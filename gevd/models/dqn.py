import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from gevd.models.networks import QNetwork


class DQNAgent:
    """Masked target-network Q primitive shared by N=1 DQN and VDN.

    For joint replay, local utilities are summed before one team TD loss; no
    per-robot copy of the team target is fitted.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        learning_rate,
        gamma,
        target_update_interval,
        gradient_clip,
        seed,
        device="cpu",
        network=None,
        loss_kind="huber",
        td_reward_scale=1.0,
        double_dqn=False,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive.")
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.gradient_clip = None if gradient_clip is None else float(gradient_clip)
        if self.gradient_clip is not None and self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive or null.")
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        if loss_kind not in {"huber", "mse"}:
            raise ValueError("loss_kind must be 'huber' or 'mse'.")
        self.loss_kind = loss_kind
        if float(td_reward_scale) <= 0.0:
            raise ValueError("td_reward_scale must be positive.")
        self.td_reward_scale = float(td_reward_scale)
        if not isinstance(double_dqn, bool):
            raise TypeError("double_dqn must be boolean.")
        self.double_dqn = double_dqn
        self.online = (
            QNetwork(obs_dim, hidden_dim, action_dim) if network is None else network
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(learning_rate))
        self.update_steps = 0
        self.last_predicted_q = np.empty(0, dtype=np.float32)
        self.last_td_targets = np.empty(0, dtype=np.float32)
        self.last_td_errors = np.empty(0, dtype=np.float32)
        self.last_gradient_norm = None
        self.last_vdn_loss = None
        self.last_auxiliary_loss = 0.0
        self.last_target_sync_update = None
        self.target_sync_steps = []
        self.diagnostic_history = []

    @staticmethod
    def _mask_tensor(mask, device):
        if isinstance(mask, torch.Tensor):
            return mask.to(device=device, dtype=torch.bool)
        array = np.asarray(mask, dtype=np.bool_)
        if not array.flags.writeable:
            array = array.copy()
        return torch.as_tensor(array, dtype=torch.bool, device=device)

    def select_action(self, observation, invalid_action_mask, epsilon=0.0):
        actions = self.select_actions(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            np.asarray(invalid_action_mask, dtype=np.bool_).reshape(1, -1),
            epsilon=epsilon,
        )
        return int(actions[0])

    def select_actions(
        self,
        observations,
        invalid_action_masks,
        epsilon=0.0,
        network=None,
        rng=None,
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        observation_array = np.asarray(observations, dtype=np.float32)
        if not observation_array.flags.writeable:
            observation_array = observation_array.copy()
        observation_tensor = torch.as_tensor(
            observation_array, dtype=torch.float32, device=self.device
        )
        invalid_masks = self._mask_tensor(invalid_action_masks, self.device)
        if observation_tensor.ndim != 2 or observation_tensor.shape[1] != self.obs_dim:
            raise ValueError("Local observations must be shaped [robots, obs_dim].")
        if invalid_masks.shape != (observation_tensor.shape[0], self.action_dim):
            raise ValueError(
                "Local action masks must be shaped [robots, action_dim]."
            )
        if ((~invalid_masks).sum(dim=1) == 0).any():
            raise ValueError("A nonterminal robot observation has no valid action.")
        behavior_network = self.online if network is None else network
        behavior_rng = self.rng if rng is None else rng
        with torch.no_grad():
            q_values = behavior_network(observation_tensor)
            q_values = q_values.masked_fill(invalid_masks, -torch.inf)
        actions = []
        for robot in range(observation_tensor.shape[0]):
            valid_actions = torch.where(~invalid_masks[robot])[0].tolist()
            if epsilon > 0.0 and behavior_rng.random() < epsilon:
                actions.append(int(behavior_rng.choice(valid_actions)))
            else:
                actions.append(int(torch.argmax(q_values[robot]).item()))
        return np.asarray(actions, dtype=np.int64)

    def compute_td_targets(self, rewards, next_states, next_invalid_masks, dones):
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1)
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        next_invalid_masks = self._mask_tensor(next_invalid_masks, self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device).reshape(-1)
        if next_states.ndim == 2:
            next_states = next_states.unsqueeze(1)
        if next_invalid_masks.ndim == 2:
            next_invalid_masks = next_invalid_masks.unsqueeze(1)
        if next_states.ndim != 3 or next_invalid_masks.ndim != 3:
            raise ValueError("Joint successor observations and masks must be rank-3.")
        if next_states.shape[0] != rewards.shape[0] or next_invalid_masks.shape[0] != rewards.shape[0]:
            raise ValueError("TD target batch dimensions do not match.")
        if dones.shape[0] != rewards.shape[0]:
            raise ValueError("TD target done flags do not match the batch size.")
        if next_states.shape[2] != self.obs_dim:
            raise ValueError("Successor observation width does not match the Q network.")
        if next_invalid_masks.shape[1] != next_states.shape[1]:
            raise ValueError("Successor robot dimensions do not match.")
        if next_invalid_masks.shape[2] != self.action_dim:
            raise ValueError("Successor action-mask width does not match the Q network.")

        next_values = torch.zeros_like(rewards)
        active = ~dones
        if active.any():
            active_masks = next_invalid_masks[active]
            if ((~active_masks).sum(dim=2) == 0).any():
                raise ValueError("A nonterminal successor has no valid action.")
            with torch.no_grad():
                active_states = next_states[active]
                active_q = self.target(
                    active_states.reshape(-1, self.obs_dim)
                ).reshape(active_states.shape[0], active_states.shape[1], self.action_dim)
                active_q = active_q.masked_fill(active_masks, -torch.inf)
                if self.double_dqn:
                    # Online utilities select legal actions; frozen target
                    # utilities evaluate them. VDN still sums once per team.
                    selection_q = self.online(
                        active_states.reshape(-1, self.obs_dim)
                    ).reshape(active_states.shape[0], active_states.shape[1], self.action_dim)
                    selected = selection_q.masked_fill(active_masks, -torch.inf).argmax(dim=2)
                    next_values[active] = active_q.gather(
                        2, selected.unsqueeze(2)
                    ).squeeze(2).sum(dim=1)
                else:
                    next_values[active] = active_q.max(dim=2).values.sum(dim=1)
        return self.td_reward_scale * rewards + self.gamma * next_values

    def learn(self, batch, auxiliary_loss=None):
        """Apply one unchanged VDN TD update plus an optional external loss.

        ``auxiliary_loss`` is deliberately computed outside this learner.  It
        may influence the shared local utility network, but it cannot replace
        or rewrite the team reward, TD target, or additive VDN summation.
        """
        states = torch.as_tensor(
            np.stack(batch.state), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(np.asarray(batch.action), dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack(batch.next_state), dtype=torch.float32, device=self.device
        )
        invalid_masks = self._mask_tensor(np.stack(batch.action_mask), self.device)
        next_invalid_masks = self._mask_tensor(
            np.stack(batch.next_action_mask), self.device
        )
        dones = torch.as_tensor(batch.done, dtype=torch.bool, device=self.device)

        if states.ndim == 2:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        if invalid_masks.ndim == 2:
            invalid_masks = invalid_masks.unsqueeze(1)
            next_invalid_masks = next_invalid_masks.unsqueeze(1)
        if states.ndim != 3 or next_states.shape != states.shape:
            raise ValueError("Replay observations must be shaped [batch, robots, obs_dim].")
        if states.shape[2] != self.obs_dim:
            raise ValueError("Replay observation width does not match the Q network.")
        if actions.ndim != 2 or actions.shape != states.shape[:2]:
            raise ValueError("Replay actions do not match the transition batch.")
        if ((actions < 0) | (actions >= self.action_dim)).any():
            raise ValueError("Replay contains an out-of-range action.")
        expected_mask_shape = (states.shape[0], states.shape[1], self.action_dim)
        if invalid_masks.shape != expected_mask_shape:
            raise ValueError("Replay action masks do not match the Q-network output.")
        if next_invalid_masks.shape != expected_mask_shape:
            raise ValueError("Replay successor masks do not match the Q-network output.")
        batch_index = torch.arange(actions.shape[0], device=self.device).unsqueeze(1)
        robot_index = torch.arange(actions.shape[1], device=self.device).unsqueeze(0)
        if invalid_masks[batch_index, robot_index, actions].any():
            raise ValueError("Replay contains an invalid executed action.")

        local_q = self.online(states.reshape(-1, self.obs_dim)).reshape(
            states.shape[0], states.shape[1], self.action_dim
        )
        predicted = local_q.gather(2, actions.unsqueeze(2)).squeeze(2).sum(dim=1)
        targets = self.compute_td_targets(rewards, next_states, next_invalid_masks, dones)
        td_errors = targets - predicted
        vdn_loss = (
            F.smooth_l1_loss(predicted, targets)
            if self.loss_kind == "huber"
            else torch.mean(td_errors.square())
        )
        if auxiliary_loss is None:
            auxiliary = torch.zeros((), dtype=vdn_loss.dtype, device=self.device)
        else:
            auxiliary = auxiliary_loss
            if not isinstance(auxiliary, torch.Tensor) or auxiliary.numel() != 1:
                raise TypeError("auxiliary_loss must be a scalar torch tensor.")
            auxiliary = auxiliary.to(device=self.device, dtype=vdn_loss.dtype)
            if not bool(torch.isfinite(auxiliary).item()) or float(auxiliary.detach()) < 0.0:
                raise FloatingPointError("auxiliary_loss must be finite and non-negative.")
        loss = vdn_loss + auxiliary
        self.last_vdn_loss = float(vdn_loss.detach().cpu().item())
        self.last_auxiliary_loss = float(auxiliary.detach().cpu().item())
        self.last_predicted_q = predicted.detach().cpu().numpy().copy()
        self.last_td_targets = targets.detach().cpu().numpy().copy()
        self.last_td_errors = td_errors.detach().cpu().numpy().copy()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.gradient_clip is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.online.parameters(), self.gradient_clip
            )
            self.last_gradient_norm = float(gradient_norm.detach().cpu().item())
        else:
            squared_norm = torch.zeros((), device=self.device)
            for parameter in self.online.parameters():
                if parameter.grad is not None:
                    squared_norm += parameter.grad.detach().square().sum()
            self.last_gradient_norm = float(squared_norm.sqrt().cpu().item())
        self.optimizer.step()

        gradient_devices = {
            parameter.grad.device
            for parameter in self.online.parameters()
            if parameter.grad is not None
        }
        model_device = next(self.online.parameters()).device
        if gradient_devices != {model_device}:
            raise RuntimeError(
                f"Expected all gradients on {model_device}, got {sorted(map(str, gradient_devices))}."
            )
        self.last_gradient_device = str(model_device)

        self.update_steps += 1
        self.last_target_sync_update = None
        if self.update_steps % self.target_update_interval == 0:
            self.sync_target()
            self.last_target_sync_update = int(self.update_steps)
            self.target_sync_steps.append(int(self.update_steps))
        self.diagnostic_history.append(
            {
                "update_step": int(self.update_steps),
                "predicted_min": float(predicted.detach().min().cpu().item()),
                "predicted_max": float(predicted.detach().max().cpu().item()),
                "target_min": float(targets.detach().min().cpu().item()),
                "target_max": float(targets.detach().max().cpu().item()),
                "td_abs_mean": float(td_errors.detach().abs().mean().cpu().item()),
                "td_abs_max": float(td_errors.detach().abs().max().cpu().item()),
                "vdn_td_loss": self.last_vdn_loss,
                "event_aligned_auxiliary_loss": self.last_auxiliary_loss,
                "combined_loss": float(loss.detach().cpu().item()),
                "gradient_norm_before_clip": float(self.last_gradient_norm),
                "target_synced": self.last_target_sync_update is not None,
            }
        )
        return float(loss.detach().cpu().item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def frozen_network(self):
        snapshot = copy.deepcopy(self.online).to(self.device)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        return snapshot

    def state_dict(self):
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "seed": self.seed,
            "rng_state": self.rng.getstate(),
            "loss_kind": self.loss_kind,
            "td_reward_scale": self.td_reward_scale,
            "double_dqn": self.double_dqn,
            "target_sync_steps": list(self.target_sync_steps),
        }

    def load_state_dict(self, state):
        if 'retrospective_utility' in state:
            raise ValueError('A GEVD checkpoint requires its auxiliary branch, not a primary-only DQN.')
        if state.get("double_dqn", False) != self.double_dqn:
            raise ValueError("Checkpoint DQN target rule does not match the learner; start a fresh run.")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)
        self.update_steps = int(state["update_steps"])
        self.seed = int(state["seed"])
        if state.get("loss_kind", self.loss_kind) != self.loss_kind:
            raise ValueError("Checkpoint loss kind does not match the learner.")
        if not np.isclose(
            float(state.get("td_reward_scale", 1.0)), self.td_reward_scale
        ):
            raise ValueError("Checkpoint TD reward scale does not match the learner.")
        self.rng = random.Random(self.seed)
        self.rng.setstate(state["rng_state"])
        self.target_sync_steps = [
            int(value) for value in state.get("target_sync_steps", [])
        ]
        self.diagnostic_history = []
        self.last_target_sync_update = (
            self.target_sync_steps[-1] if self.target_sync_steps else None
        )
        self.target.eval()
