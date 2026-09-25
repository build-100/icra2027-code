"""Independent event outcome estimator; normalized credit is detached from VDN.

Counterfactual labels are simulator/reference-continuation estimates, not true
causal effects.  They are never inserted in factual or MCBR replay.
"""
from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from gevd.training.event_credit import EventAlignedCreditModule, EventCreditSample


@dataclass
class CreditGroup:
    samples: tuple[EventCreditSample, ...]
    features: tuple[np.ndarray, ...]  # per candidate: factual then alternatives
    fixed_weights: np.ndarray


class LearnedEventCreditModule(EventAlignedCreditModule):
    """Q_phi predicts bounded task outcome; Q_go - mean(Q_alternative) scores history.

    The reference continuation follows a factual local action if the local
    history at that time is identical to the recorded history, otherwise a
    frozen greedy policy. Both interventions use this same continuation rule.
    The factual intervention therefore reproduces the successful witness;
    alternatives are allowed to recover or achieve another successful route.
    """

    def __init__(self, config: Mapping[str, Any], seed: int, env: Any) -> None:
        super().__init__(config, seed)
        if self.aligner.qualification != "terminal_success_chain":
            raise ValueError("Learned credit requires terminal_success_chain.")
        if self.aligner.event_classes != ("E3",):
            raise ValueError("The minimal learned candidate supports E3 only.")
        if self.aligner.trace_rule != "causal_prefix_decay":
            raise ValueError("Learned candidates require the endpoint historical prefix.")
        self.mode = str(config.get("mode", "learned"))
        if self.mode not in {"learned", "fixed_normalized"}:
            raise ValueError("Unsupported normalized event mode.")
        settings = dict(config.get("learned", {}))
        self.total_credit = float(settings.get("total_credit", 3.0))
        self.temperature = float(settings.get("temperature", 0.25))
        self.uniform_floor = float(settings.get("uniform_floor", 0.1))
        self.max_roots = int(settings.get("max_roots_per_episode", 6))
        self.max_alternatives = int(settings.get("max_alternatives", 2))
        self.label_interval = int(settings.get("label_interval_successes", 4))
        self.max_sim_steps = int(settings.get("max_counterfactual_steps", 16000))
        self.updates_per_episode = int(settings.get("updates_per_qualified_episode", 4))
        self.update_only_with_new_labels = settings.get("update_only_with_new_labels", False)
        if not isinstance(self.update_only_with_new_labels, bool):
            raise TypeError("update_only_with_new_labels must be boolean")
        self.critic_skipped_no_new_labels = 0
        self.group_capacity = int(settings.get("group_capacity", 1000))
        self.label_capacity = int(settings.get("label_capacity", 8000))
        self.rng = random.Random(seed + 101)
        self.train_rng = random.Random(seed + 103)
        self.sample_rng = random.Random(seed + 107)
        if not np.isfinite(self.total_credit) or self.total_credit <= 0:
            raise ValueError("total_credit must be finite and positive.")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not 0 <= self.uniform_floor < 1:
            raise ValueError("uniform_floor must be in [0, 1).")
        if min(self.max_roots, self.max_alternatives, self.label_interval,
               self.max_sim_steps, self.updates_per_episode, self.group_capacity,
               self.label_capacity) < 1:
            raise ValueError("Learned credit budgets and capacities must be positive.")
        self.groups: list[CreditGroup] = []
        self.labels: list[tuple[np.ndarray, np.ndarray]] = []
        self.group_position = self.label_position = 0
        # Central history is used only for training supervision, never execution.
        self.feature_dim = env.num_robots * env.observation_dim + env.num_robots + 2 * env.num_nodes + 8
        hidden = int(settings.get("hidden_dim", 64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 109)
            self.network = nn.Sequential(nn.Linear(self.feature_dim, hidden), nn.ReLU(),
                                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(settings.get("learning_rate", 0.001)))
        self.counterfactual_steps = 0
        self.counterfactual_rollouts = 0
        self.labeled_candidates = 0
        self.critic_updates = 0
        self.critic_seconds = 0.0
        self.simulator_seconds = 0.0
        self.critic_loss = None
        self.last_groups: list[dict[str, Any]] = []
        self.label_audit: list[dict[str, Any]] = []

    def _features(self, sample: EventCreditSample, source: Any, actions: Sequence[int], env: Any) -> np.ndarray:
        obs = np.asarray(source.observations, dtype=np.float32).copy()
        obs[:, :env.num_edges] /= env.t_max
        obs[:, -1] /= env.t_max
        robot = np.eye(env.num_robots, dtype=np.float32)[sample.robot]
        region = np.eye(env.num_nodes, dtype=np.float32)[sample.region_index]
        position = int(source.state.positions[sample.robot])
        rows = []
        for action in actions:
            destination = int(env.neighbor_node_indices[position, action])
            if destination < 0:
                raise ValueError("Counterfactual action is invalid at the historical state.")
            dest = np.eye(env.num_nodes, dtype=np.float32)[destination]
            scalars = np.asarray([
                sample.event_time / env.t_max, sample.action_time / env.t_max,
                sample.delay / env.t_max,
                float(sample.role == "initiator"), float(sample.role == "completer"),
                float(sample.role == "enabling_trace"), float(destination == sample.region_index),
                float(not source.state.O[sample.robot, destination]),
            ], dtype=np.float32)
            rows.append(np.concatenate((obs.ravel(), robot, region, dest, scalars)))
        return np.stack(rows)

    def _rollout(self, sample: EventCreditSample, action: int, cache: Sequence[Any],
                 env: Any, agent: Any, frozen: nn.Module) -> dict[str, Any]:
        source = cache[sample.action_time]
        observations, masks = env.restore(source.state)
        joint = list(source.factual_action)
        joint[sample.robot] = int(action)
        distance = 0.0
        first = True
        steps = 0
        while not env.done:
            if not first:
                joint = list(agent.select_actions(observations, masks, epsilon=0.0, network=frozen))
                t = env.state.time
                if t < len(cache):
                    # A fixed reference-conditioned continuation, applied to both arms.
                    for robot in range(env.num_robots):
                        if np.array_equal(observations[robot], cache[t].observations[robot]):
                            joint[robot] = cache[t].factual_action[robot]
            result = env.step(joint)
            first = False
            distance += result.travel_distance
            steps += 1
            observations, masks = result.observations, result.action_masks
        coverage = env.coverage_count() / env.num_nodes
        components = env.component_count()
        fused = float(components == 1)
        distance_scale = env.num_robots * env.t_max * max(float(env.edge_distances.max()), 1e-8)
        # Strict success dominates all bounded partial-progress and cost terms.
        outcome = (float(env.success) + 0.25 * coverage + 0.125 * fused
                   - 0.05 * env.state.time / env.t_max - 0.05 * distance / distance_scale)
        self.counterfactual_steps += steps
        self.counterfactual_rollouts += 1
        return {"outcome": outcome, "success": bool(env.success), "coverage": coverage,
                "components": components, "steps": steps, "end_time": env.state.time,
                "continuation_distance": distance}

    def _label(self, candidates: list[tuple[EventCreditSample, np.ndarray, list[int]]],
               cache: Sequence[Any], env: Any, agent: Any) -> None:
        remaining = self.max_sim_steps - self.counterfactual_steps
        if self.mode != "learned" or remaining <= 0:
            return
        snapshot = env.snapshot()
        frozen = agent.frozen_network()
        started = time.perf_counter()
        try:
            for sample, features, actions in self.rng.sample(candidates, min(self.max_roots, len(candidates))):
                bound = len(actions) * (env.t_max - sample.action_time)
                if self.counterfactual_steps + bound > self.max_sim_steps:
                    continue
                outcomes = [self._rollout(sample, action, cache, env, agent, frozen) for action in actions]
                if not outcomes[0]["success"]:
                    raise AssertionError("The factual intervention did not reproduce its success witness.")
                targets = np.asarray([item["outcome"] for item in outcomes], dtype=np.float32)
                entry = (features.copy(), targets)
                if len(self.labels) < self.label_capacity:
                    self.labels.append(entry)
                else:
                    self.labels[self.label_position] = entry
                self.label_position = (self.label_position + 1) % self.label_capacity
                self.labeled_candidates += 1
                audit = {"event_id": sample.event_id, "robot": sample.robot,
                         "action_time": sample.action_time, "event_time": sample.event_time,
                         "actions": actions, "outcomes": outcomes,
                         "contribution_proxy": float(targets[0] - targets[1:].mean())}
                if len(self.label_audit) < 100:
                    self.label_audit.append(audit)
        finally:
            env.restore(snapshot)
            self.simulator_seconds += time.perf_counter() - started
        if not env.state.equals(snapshot):
            raise AssertionError("Event labeling changed the factual environment state.")

    def _train_critic(self) -> None:
        if self.mode != "learned" or not self.labels:
            return
        started = time.perf_counter()
        for _ in range(self.updates_per_episode):
            batch = self.train_rng.sample(self.labels, min(32, len(self.labels)))
            x = torch.from_numpy(np.concatenate([entry[0] for entry in batch]))
            y = torch.from_numpy(np.concatenate([entry[1] for entry in batch]))
            predictions = self.network(x).squeeze(-1)
            loss = (predictions - y).square().mean()
            # Paired differences anchor contribution as well as absolute outcome.
            cursor = 0
            differences = []
            for features, targets in batch:
                size = len(targets)
                differences.append((predictions[cursor] - predictions[cursor + 1:cursor + size].mean()
                                    - float(targets[0] - targets[1:].mean())).square())
                cursor += size
            loss = loss + torch.stack(differences).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite event outcome loss.")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.critic_loss = float(loss.detach())
            self.critic_updates += 1
        self.critic_seconds += time.perf_counter() - started

    def weights(self, group: CreditGroup) -> np.ndarray:
        if self.mode == "fixed_normalized":
            return group.fixed_weights.copy()
        if not self.critic_updates:
            return np.full(len(group.samples), 1.0 / len(group.samples))
        with torch.no_grad():
            values = self.network(torch.from_numpy(np.concatenate(group.features))).squeeze(-1).numpy()
        cursor = 0
        advantages = []
        for features in group.features:
            size = len(features)
            advantages.append(max(0.0, float(values[cursor] - values[cursor + 1:cursor + size].mean())))
            cursor += size
        scores = np.asarray(advantages, dtype=np.float64) / self.temperature
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        weights = (1 - self.uniform_floor) * weights + self.uniform_floor / len(weights)
        return weights

    def add_episode(self, factual_cache: Sequence[Any], transitions: Sequence[Any],
                    *, env: Any, agent: Any) -> tuple[EventCreditSample, ...]:
        if not self.enabled:
            return ()
        self.observed_episode_count += 1
        aligned = self.aligner.align_episode(factual_cache, transitions)
        grouped: dict[str, list[EventCreditSample]] = defaultdict(list)
        for sample in aligned:
            if int((~sample.invalid_action_mask).sum()) > 1:
                grouped[sample.event_id].append(sample)
        if not grouped:
            return ()
        self.qualified_episode_count += 1
        self.event_occurrence_count += len(grouped)
        groups = []
        candidates = []
        for event_id, samples in grouped.items():
            matrices = []
            for sample in samples:
                source = factual_cache[sample.action_time]
                alternatives = [int(a) for a in np.flatnonzero(~sample.invalid_action_mask) if a != sample.action]
                # Deterministic evenly spaced subset avoids random consumption in the actor.
                alternatives = alternatives[:self.max_alternatives]
                actions = [sample.action, *alternatives]
                features = self._features(sample, source, actions, env)
                matrices.append(features)
                candidates.append((sample, features, actions))
            fixed = np.asarray([sample.credit for sample in samples], dtype=np.float64)
            fixed /= fixed.sum()
            groups.append(CreditGroup(tuple(samples), tuple(matrices), fixed))
            self.event_ids.add(event_id)
        labels_before = self.labeled_candidates
        if (self.qualified_episode_count - 1) % self.label_interval == 0:
            self._label(candidates, factual_cache, env, agent)
        if not self.update_only_with_new_labels or self.labeled_candidates > labels_before:
            self._train_critic()
        else:
            self.critic_skipped_no_new_labels += 1
        assignments = []
        self.last_groups = []
        for group in groups:
            if len(self.groups) < self.group_capacity:
                self.groups.append(group)
            else:
                self.groups[self.group_position] = group
            self.group_position = (self.group_position + 1) % self.group_capacity
            weights = self.weights(group)
            credits = self.total_credit * weights
            if not np.isclose(credits.sum(), self.total_credit, atol=1e-10):
                raise AssertionError("Learned credit mass is not conserved.")
            trace = []
            for sample, weight, credit in zip(group.samples, weights, credits):
                weighted = replace(sample, credit=float(credit))
                assignments.append(weighted)
                self.replay.push(weighted)
                self.assignment_count += 1
                self.role_counts[sample.role] = self.role_counts.get(sample.role, 0) + 1
                trace.append({"robot": sample.robot, "action_time": sample.action_time,
                              "event_time": sample.event_time, "role": sample.role,
                              "weight": float(weight), "credit": float(credit)})
            self.last_groups.append({"event_id": group.samples[0].event_id,
                                     "total_credit": float(credits.sum()), "assignments": trace})
        return tuple(assignments)

    def auxiliary_loss(self, network: nn.Module, device: torch.device) -> torch.Tensor | None:
        if not self.enabled or not self.groups or self.loss_weight == 0:
            return None
        # Rescore entire events with current detached Q_phi; never use stale credit.
        selected = []
        while len(selected) < self.batch_size:
            group = self.sample_rng.choice(self.groups)
            weights = self.weights(group)
            selected.extend(replace(s, credit=float(self.total_credit * w))
                            for s, w in zip(group.samples, weights))
        samples = self.sample_rng.sample(selected, min(self.batch_size, len(selected)))
        obs = torch.as_tensor(np.stack([s.observation for s in samples]), device=device)
        invalid = torch.as_tensor(np.stack([s.invalid_action_mask for s in samples]), device=device)
        actions = torch.as_tensor([s.action for s in samples], device=device)
        credits = torch.as_tensor([s.credit for s in samples], device=device, dtype=torch.float32)
        q = network(obs)
        row = torch.arange(len(samples), device=device)
        alternatives = invalid.clone()
        alternatives[row, actions] = True
        best = q.masked_fill(alternatives, -torch.inf).max(dim=1).values
        raw = torch.relu(self.margin_scale * credits - (q[row, actions] - best)).square().mean()
        self.last_diagnostics = {"sample_count": len(samples), "usable_count": len(samples),
                                 "raw_loss": float(raw.detach()),
                                 "weighted_loss": float(self.loss_weight * raw.detach()),
                                 "mean_credit": float(credits.mean()),
                                 "mean_delay": float(np.mean([s.delay for s in samples])),
                                 "max_delay": max(s.delay for s in samples)}
        return self.loss_weight * raw

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        result.update({"mode": self.mode, "total_credit_per_event": self.total_credit,
                       "update_only_with_new_labels": self.update_only_with_new_labels,
                       "critic_skipped_no_new_labels": self.critic_skipped_no_new_labels,
                       "fixed_total_credit": True, "credit_gradient_from_vdn": False,
                       "counterfactual_steps": self.counterfactual_steps,
                       "counterfactual_rollouts": self.counterfactual_rollouts,
                       "labeled_candidates": self.labeled_candidates,
                       "critic_updates": self.critic_updates, "critic_loss": self.critic_loss,
                       "critic_seconds": self.critic_seconds, "simulator_seconds": self.simulator_seconds,
                       "critic_parameter_count": sum(p.numel() for p in self.network.parameters()),
                       "group_replay_size": len(self.groups), "last_groups": self.last_groups,
                       "counterfactual_continuation": "factual_local_history_else_frozen_greedy",
                       "contribution_claim": "simulator_and_reference_continuation_proxy"})
        return result

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["learned_state"] = {"network": copy.deepcopy(self.network.state_dict()),
                                   "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                                   "groups": self.groups, "labels": self.labels,
                                   "rng": self.rng.getstate(), "train_rng": self.train_rng.getstate(),
                                   "sample_rng": self.sample_rng.getstate(), "mode": self.mode}
        for name in ("group_position", "label_position", "counterfactual_steps", "counterfactual_rollouts",
                     "critic_skipped_no_new_labels",
                     "labeled_candidates", "critic_updates", "critic_loss", "critic_seconds",
                     "simulator_seconds", "last_groups", "label_audit"):
            result["learned_state"][name] = getattr(self, name)
        return result

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        learned = state["learned_state"]
        if learned["mode"] != self.mode:
            raise ValueError("Event credit mode differs from checkpoint.")
        self.network.load_state_dict(learned["network"])
        self.optimizer.load_state_dict(learned["optimizer"])
        self.rng.setstate(learned["rng"])
        self.train_rng.setstate(learned["train_rng"])
        self.sample_rng.setstate(learned["sample_rng"])
        for name, value in learned.items():
            if name not in {"network", "optimizer", "rng", "train_rng", "sample_rng", "mode"}:
                setattr(self, name, value)


def make_event_credit(config: Mapping[str, Any], seed: int, env: Any, *, reward_scale: float = 1.0) -> EventAlignedCreditModule:
    mode = str(config.get("mode", "fixed"))
    if mode not in {"fixed", "learned", "fixed_normalized", "paired_return"}:
        raise ValueError(f"Unknown event credit mode: {mode}")
    if mode == "fixed" or not bool(config.get("enabled", False)):
        return EventAlignedCreditModule(config, seed)
    if mode == "paired_return":
        from gevd.training.paired_return_credit import PairedReturnCreditModule
        return PairedReturnCreditModule(config, seed, env, reward_scale)
    return LearnedEventCreditModule(config, seed, env)
