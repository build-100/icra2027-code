"""Independent, bounded auxiliary utility trained on factual Retrace segments.

The primary VDN learner is delegated to without changing its reward or TD rule.
The auxiliary output is Q' = 0.1 * stopgrad(abs(Q)) * tanh(z), in the same
internal units as Q. The execution multiplier is separate from this hard cap.
Retrace convention: https://arxiv.org/abs/1606.02647, Eq. (3).
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from gevd.models.networks import relu_orthogonal_init_, weights_init_


ABSOLUTE_QPRIME_RATIO = 0.1
DEFAULTS = dict(enabled=True, execution_lambda=0.1, trace_decay=0.9,
                batch_size=8, replay_episodes=256, max_trace_steps=23,
                event_sample_fraction=0.5)


def resolve_settings(settings):
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown retrospective utility settings: {sorted(unknown)}")
    result = {**DEFAULTS, **settings}
    if result['enabled'] is not True:
        raise ValueError('GEVD requires retrospective_utility.enabled=true.')
    for key in ('execution_lambda', 'trace_decay', 'event_sample_fraction'):
        value = result[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError(f'{key} must be finite and in [0, 1].')
    for key in ('batch_size', 'replay_episodes', 'max_trace_steps'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer.')
    return result


def bounded_utility(primary, latent):
    """Per state/robot/action cap, including negative Q and exactly zero Q.

    No epsilon floor, batch maximum, lambda division, or gradient into Q.
    z is only a latent network output; the bounded value is the actual Q'.
    """
    if primary.shape != latent.shape:
        raise ValueError('Primary and auxiliary utility shapes differ.')
    return (ABSOLUTE_QPRIME_RATIO * primary.detach().abs()) * torch.tanh(latent)


def masked_policy(scores, invalid, epsilon=0.0):
    """The same first-argmax tie rule as the primary masked epsilon-greedy."""
    valid = ~invalid
    count = valid.sum(-1, keepdim=True)
    greedy = scores.masked_fill(invalid, -torch.inf).argmax(-1, keepdim=True)
    probabilities = epsilon * valid.to(scores.dtype) / count.clamp_min(1)
    probabilities = probabilities.scatter_add(-1, greedy,
        torch.full_like(greedy, 1.0 - epsilon, dtype=scores.dtype))
    return probabilities * valid


def truncated_retrace(rewards, q_taken, next_values, coefficients, dones, valid, gamma=1.0):
    """Batched [segments,time] targets; c[t+1], never c[t], weights continuation.

    A truncated nonterminal tail bootstraps from its true successor. Terminal
    tails bootstrap zero. coefficients use the JOINT target/behavior ratio.
    """
    target = torch.zeros_like(rewards)
    for t in range(rewards.shape[1] - 1, -1, -1):
        active = (~dones[:, t]).to(rewards.dtype)
        value = rewards[:, t] + gamma * active * next_values[:, t]
        if t + 1 < rewards.shape[1]:
            value = value + gamma * active * coefficients[:, t + 1] * valid[:, t + 1] * (
                target[:, t + 1] - q_taken[:, t + 1])
        target[:, t] = torch.where(valid[:, t], value, 0.0)
    return target.detach()


@dataclass
class FactualAuxEpisode:
    observations: np.ndarray       # T+1,N,D
    invalid_masks: np.ndarray      # T+1,N,A
    actions: np.ndarray            # T,N
    rewards: np.ndarray            # T,N, unscaled auxiliary reward at closure time
    dones: np.ndarray              # T; budget truncation is not terminal
    behavior_log_prob: np.ndarray  # T, exact joint behavior probability in log form
    event_roots: tuple             # (historical action time, robot, closure-exclusive end)


def collect_auxiliary_episode(cache, transitions, behavior_log_prob):
    """Credit only an earlier first arrival enabling a teammate's E3 closure.

    Initial occupants have no historical action. Simultaneous arrivals have no
    earlier action. Each unique E3 factor contributes 1/(N-1) once; revisits,
    E2 closures and the later completer receive no retrospective credit.
    Failed and non-fusing episodes are retained as factual negative samples.
    """
    if not cache or len(cache) != len(transitions) or len(cache) != len(behavior_log_prob):
        raise ValueError('One behavior probability and pre-action state are required per move.')
    steps, robots = len(cache), len(cache[0].factual_action)
    rewards = np.zeros((steps, robots), dtype=np.float32)
    first_arrivals = {}
    roots, seen = [], set()
    for t, (before, after) in enumerate(zip(cache, transitions)):
        if before.absolute_time != t:
            raise ValueError('Factual history must start at zero and be contiguous.')
        for robot, region in np.argwhere(after.state.O & ~before.state.O):
            first_arrivals.setdefault((int(robot), int(region)), t)
        for event in after.events:
            if event.factor_type != 'closure' or event.event_class != 'E3':
                continue
            if event.robot_pair is None or event.region_index is None or robots < 2:
                raise ValueError('An E3 closure requires two robot endpoints and a region.')
            key = (tuple(sorted(event.robot_pair)), int(event.region_index))
            if key in seen:
                continue
            seen.add(key)
            earlier = [(robot, first_arrivals.get((robot, int(event.region_index))))
                       for robot in event.robot_pair]
            earlier = [(robot, time) for robot, time in earlier if time is not None and time < t]
            if len(earlier) > 1:
                raise ValueError('A newly activated star closure cannot have two earlier arrivals.')
            for robot, time in earlier:
                rewards[t, robot] += 1.0 / (robots - 1)
                roots.append((time, robot, t + 1))
    probabilities = np.asarray(behavior_log_prob, dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities > 1e-12).any():
        raise ValueError('Executed behavior probabilities must be finite and positive.')
    return FactualAuxEpisode(
        np.stack([x.observations for x in cache] + [transitions[-1].observations]),
        np.stack([x.action_masks for x in cache] + [
            np.ones_like(transitions[-1].action_masks) if transitions[-1].done
            else transitions[-1].action_masks]),
        np.asarray([x.factual_action for x in cache], dtype=np.int64), rewards,
        np.asarray([x.done for x in transitions], dtype=bool), probabilities, tuple(roots))


class RetrospectiveUtilityAgent:
    """Wrap the unchanged primary DQN with a separately optimized auxiliary branch."""
    def __init__(self, primary, settings):
        self.primary = primary
        self.settings = resolve_settings(settings)
        # Independent initialization must not consume the primary run's RNG stream.
        self.auxiliary = copy.deepcopy(primary.online).cpu()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(primary.seed + 17011)
            initializer = (relu_orthogonal_init_ if
                getattr(self.auxiliary, 'initialization', '') == 'relu_orthogonal' else weights_init_)
            self.auxiliary.apply(initializer)
            head = self.auxiliary.utility_head[-1]
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.auxiliary.to(primary.device)
        self.auxiliary_target = copy.deepcopy(self.auxiliary).eval()
        self.auxiliary_optimizer = torch.optim.Adam(self.auxiliary.parameters(),
            lr=primary.optimizer.param_groups[0]['lr'])
        self.auxiliary_rng = random.Random(primary.seed + 17027)
        self.auxiliary_replay = []
        self.replay_position = 0
        self.auxiliary_updates = 0
        self.auxiliary_episodes = 0
        self.auxiliary_events = 0
        self.capture = None

    def __getattr__(self, name):
        return getattr(self.primary, name)

    def execution_values(self, observations):
        primary = self.online(observations)
        auxiliary = bounded_utility(primary, self.auxiliary(observations))
        return primary, auxiliary, primary + self.settings['execution_lambda'] * auxiliary

    def select_action(self, observation, invalid_action_mask, epsilon=0.0):
        return int(self.select_actions(np.asarray(observation, dtype=np.float32).reshape(1, -1),
            np.asarray(invalid_action_mask, dtype=bool).reshape(1, -1), epsilon=epsilon)[0])

    def select_actions(self, observations, invalid_action_masks, epsilon=0.0, network=None, rng=None):
        if network is not None:
            raise ValueError('GEVD uses factual combined-policy actions, not paired search.')
        values = {}
        def combined(inputs):
            q, qp, score = self.execution_values(inputs)
            values.update(q=q, qp=qp, score=score)
            return score
        actions = self.primary.select_actions(observations, invalid_action_masks,
            epsilon=epsilon, network=combined, rng=rng)
        if self.capture is not None:
            masks = self._mask_tensor(invalid_action_masks, self.device)
            probabilities = masked_policy(values['score'].double(), masks, epsilon)
            executed = probabilities.gather(1, torch.as_tensor(actions, device=self.device)[:, None])
            log_mu = float(executed.double().log().sum().cpu())
            valid = ~masks
            q, qp = values['q'][valid], values['qp'][valid]
            denominator = torch.where(q != 0, q.abs(), torch.ones_like(q))
            ratio = qp.abs() / denominator
            if (qp.abs() > ABSOLUTE_QPRIME_RATIO * q.abs()).any():
                raise AssertionError('Q-prime exceeded its absolute per-action bound.')
            primary_greedy = values['q'].masked_fill(masks, -torch.inf).argmax(-1)
            combined_greedy = values['score'].masked_fill(masks, -torch.inf).argmax(-1)
            self.capture.append(dict(log_mu=log_mu, max_ratio=float(ratio.max().cpu()),
                greedy_changes=int((primary_greedy != combined_greedy).sum().cpu())))
        return actions

    def add_episode(self, cache, transitions, log_probs):
        episode = collect_auxiliary_episode(cache, transitions, log_probs)
        if len(self.auxiliary_replay) < self.settings['replay_episodes']:
            self.auxiliary_replay.append(episode)
        else:
            self.auxiliary_replay[self.replay_position] = episode
        self.replay_position = (self.replay_position + 1) % self.settings['replay_episodes']
        self.auxiliary_episodes += 1
        self.auxiliary_events += len(episode.event_roots)
        return len(episode.event_roots)

    def _sample_segments(self):
        episodes = self.auxiliary_replay
        roots = [(index, root) for index, episode in enumerate(episodes) for root in episode.event_roots]
        weights = [episode.actions.size for episode in episodes]
        samples = []
        for _ in range(self.settings['batch_size']):
            if roots and self.auxiliary_rng.random() < self.settings['event_sample_fraction']:
                index, (start, robot, end) = self.auxiliary_rng.choice(roots)
                episode = episodes[index]
                if end - start > self.settings['max_trace_steps']:
                    raise ValueError('max_trace_steps must cover the full historical-to-closure segment.')
            else:
                episode = self.auxiliary_rng.choices(episodes, weights=weights, k=1)[0]
                start = self.auxiliary_rng.randrange(len(episode.actions))
                robot = self.auxiliary_rng.randrange(episode.actions.shape[1])
                end = min(len(episode.actions), start + self.settings['max_trace_steps'])
            samples.append((episode, start, end, robot))
        return samples

    def learn_auxiliary(self):
        if not self.auxiliary_replay:
            return None
        samples = self._sample_segments()
        batch = len(samples)
        length = max(end - start for _, start, end, _ in samples)
        robots = samples[0][0].actions.shape[1]
        states = np.zeros((batch, length + 1, robots, self.obs_dim), np.float32)
        masks = np.ones((batch, length + 1, robots, self.action_dim), bool)
        actions = np.zeros((batch, length, robots), np.int64)
        rewards = np.zeros((batch, length), np.float32)
        dones = np.ones((batch, length), bool)
        valid = np.zeros((batch, length), bool)
        log_mu = np.zeros((batch, length), np.float64)
        selected_robots = []
        for b, (episode, start, end, robot) in enumerate(samples):
            n = end - start
            states[b, :n+1] = episode.observations[start:end+1]
            masks[b, :n+1] = episode.invalid_masks[start:end+1]
            actions[b, :n] = episode.actions[start:end]
            rewards[b, :n] = episode.rewards[start:end, robot] * self.td_reward_scale
            dones[b, :n] = episode.dones[start:end]
            valid[b, :n] = True
            log_mu[b, :n] = episode.behavior_log_prob[start:end]
            selected_robots.append(robot)
        states = torch.as_tensor(states, device=self.device)
        masks = torch.as_tensor(masks, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        index = torch.arange(batch, device=self.device)
        robot_index = torch.as_tensor(selected_robots, device=self.device)
        with torch.no_grad():
            flat = states.reshape(-1, self.obs_dim)
            q, qp, combined = self.execution_values(flat)
            shape = (batch, length + 1, robots, self.action_dim)
            policy = masked_policy(combined.reshape(shape), masks, epsilon=0.0)
            target_qp = bounded_utility(self.target(flat), self.auxiliary_target(flat)).reshape(shape)
            # All robots' executed actions participate in the importance ratio.
            executed_pi = policy[:, :-1].gather(-1, actions[..., None]).squeeze(-1)
            joint_log_pi = executed_pi.double().log().sum(-1)
            log_ratio = joint_log_pi - torch.as_tensor(log_mu, device=self.device)
            c = self.settings['trace_decay'] * torch.exp(log_ratio.clamp(max=0)).float()
            local_target = target_qp[index, :, robot_index]
            local_policy = policy[index, :, robot_index]
            own_actions = actions[index, :, robot_index]
            q_taken = local_target[:, :-1].gather(-1, own_actions[..., None]).squeeze(-1)
            next_values = (local_policy[:, 1:] * local_target[:, 1:]).sum(-1)
            targets = truncated_retrace(torch.as_tensor(rewards, device=self.device), q_taken,
                next_values, c, torch.as_tensor(dones, device=self.device),
                torch.as_tensor(valid, device=self.device), self.gamma)[:, 0]
        root_states = states[index, 0, robot_index]
        root_actions = actions[index, 0, robot_index]
        with torch.no_grad():
            root_primary = self.online(root_states)
        qp_predicted = bounded_utility(root_primary, self.auxiliary(root_states))
        predicted = qp_predicted.gather(-1, root_actions[:, None]).squeeze(-1)
        loss = (predicted - targets).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite retrospective utility loss.')
        for group in self.auxiliary_optimizer.param_groups:
            group['lr'] = self.optimizer.param_groups[0]['lr']
        self.auxiliary_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.auxiliary.parameters(), self.gradient_clip, error_if_nonfinite=True)
        self.auxiliary_optimizer.step()
        self.auxiliary_updates += 1
        if self.auxiliary_updates % self.target_update_interval == 0:
            self.auxiliary_target.load_state_dict(self.auxiliary.state_dict())
        return dict(retrospective_loss=float(loss.detach().cpu()),
            retrospective_updates=self.auxiliary_updates,
            retrospective_target_abs_max=float(targets.abs().max().cpu()),
            retrospective_prediction_abs_max=float(predicted.detach().abs().max().cpu()),
            retrospective_nonzero_trace_fraction=float((c[torch.as_tensor(valid, device=self.device)] > 0).float().mean().cpu()))

    def learn(self, batch, auxiliary_loss=None):
        if auxiliary_loss is not None:
            raise ValueError('GEVD must not inject auxiliary loss into the primary encoder.')
        loss = self.primary.learn(batch)
        diagnostics = self.learn_auxiliary()
        if diagnostics:
            self.primary.diagnostic_history[-1].update(diagnostics)
        return loss

    def readonly_signature(self):
        from gevd.training.base import _network_digest
        from gevd.evaluation.replay import object_digest
        return dict(auxiliary=_network_digest(self.auxiliary), target=_network_digest(self.auxiliary_target),
            optimizer=object_digest(self.auxiliary_optimizer.state_dict()),
            replay=(len(self.auxiliary_replay), self.replay_position, self.auxiliary_episodes, self.auxiliary_events),
            rng=object_digest(self.auxiliary_rng.getstate()), updates=self.auxiliary_updates)

    def state_dict(self):
        state = self.primary.state_dict()
        state['retrospective_utility'] = dict(version=1, settings=self.settings,
            absolute_ratio=ABSOLUTE_QPRIME_RATIO, auxiliary=self.auxiliary.state_dict(),
            target=self.auxiliary_target.state_dict(), optimizer=self.auxiliary_optimizer.state_dict(),
            rng=self.auxiliary_rng.getstate(), replay=self.auxiliary_replay,
            replay_position=self.replay_position, updates=self.auxiliary_updates,
            episodes=self.auxiliary_episodes, events=self.auxiliary_events)
        return state

    def load_state_dict(self, state):
        extra = state.get('retrospective_utility')
        if not extra or extra.get('version') != 1 or extra['settings'] != self.settings or extra['absolute_ratio'] != ABSOLUTE_QPRIME_RATIO:
            raise ValueError('GEVD requires a matching auxiliary checkpoint; legacy weights cannot resume it.')
        self.primary.load_state_dict({key: value for key, value in state.items()
                                      if key != 'retrospective_utility'})
        self.auxiliary.load_state_dict(extra['auxiliary'])
        self.auxiliary_target.load_state_dict(extra['target'])
        self.auxiliary_optimizer.load_state_dict(extra['optimizer'])
        for values in self.auxiliary_optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(self.device)
        self.auxiliary_rng.setstate(extra['rng'])
        self.auxiliary_replay = copy.deepcopy(extra['replay'])
        self.replay_position = extra['replay_position']
        self.auxiliary_updates = extra['updates']
        self.auxiliary_episodes = extra['episodes']
        self.auxiliary_events = extra['events']
