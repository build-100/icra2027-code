"""Fixed-rule event-to-history credit alignment for GEVD.

The base VDN target remains the scalar team-reward TD target.  This module is
an auxiliary training path: a qualified E3 event is matched to the first
learnable arrival action of each closure endpoint, and only that robot/action
utility is encouraged to outrank the alternatives that were valid at that
historical decision.  No environment reward is changed or divided between
robots.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True, eq=False)
class EventCreditSample:
    """One precisely located local action receiving auxiliary event credit."""

    observation: np.ndarray
    action: int
    invalid_action_mask: np.ndarray
    credit: float
    event_time: int
    action_time: int
    robot: int
    region_index: int
    event_class: str
    role: str
    event_id: str

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float32).copy()
        mask = np.asarray(self.invalid_action_mask, dtype=np.bool_).copy()
        if observation.ndim != 1 or mask.ndim != 1:
            raise ValueError("Event-credit observations and masks must be rank one.")
        if not 0 <= int(self.action) < mask.shape[0] or bool(mask[int(self.action)]):
            raise ValueError("The aligned action must be valid in its historical mask.")
        if not np.isfinite(float(self.credit)) or float(self.credit) <= 0.0:
            raise ValueError("Event credit must be finite and positive.")
        if int(self.action_time) < 0 or int(self.event_time) <= int(self.action_time):
            raise ValueError("Event time must follow the indexed pre-action time.")
        observation.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "invalid_action_mask", mask)
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "credit", float(self.credit))
        object.__setattr__(self, "event_time", int(self.event_time))
        object.__setattr__(self, "action_time", int(self.action_time))
        object.__setattr__(self, "robot", int(self.robot))
        object.__setattr__(self, "region_index", int(self.region_index))

    @property
    def delay(self) -> int:
        return self.event_time - self.action_time


class EventCreditReplayBuffer:
    """Uniform replay kept separate from factual and MCBR transition replay."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if int(capacity) <= 0:
            raise ValueError("Event-credit replay capacity must be positive.")
        self.capacity = int(capacity)
        self.memory: list[EventCreditSample] = []
        self.position = 0
        self.rng = random.Random(int(seed))

    def push(self, sample: EventCreditSample) -> None:
        if not isinstance(sample, EventCreditSample):
            raise TypeError("Event-credit replay accepts EventCreditSample values only.")
        if len(self.memory) < self.capacity:
            self.memory.append(sample)
        else:
            self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[EventCreditSample, ...]:
        count = min(int(batch_size), len(self.memory))
        if count <= 0:
            return ()
        return tuple(self.rng.sample(self.memory, count))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "memory": list(self.memory),
            "position": self.position,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        capacity = int(state["capacity"])
        memory = list(state["memory"])
        position = int(state["position"])
        if capacity <= 0 or len(memory) > capacity or not 0 <= position < capacity:
            raise ValueError("Invalid event-credit replay checkpoint.")
        if not all(isinstance(item, EventCreditSample) for item in memory):
            raise TypeError("Invalid event-credit sample in checkpoint.")
        self.capacity = capacity
        self.memory = memory
        self.position = position
        self.rng.setstate(state["rng_state"])

    def __len__(self) -> int:
        return len(self.memory)


class FixedRuleEventAligner:
    """Map an E3 closure to the endpoint robots' first arrival actions.

    The fixed rule deliberately uses the ordered factual episode history rather
    than pretending that the visited-bit state contains visit timestamps.  One
    total credit is divided equally over the learnable endpoint arrivals.  An
    endpoint present in the initial state has no causal action to reinforce and
    is therefore omitted; the remaining learnable arrivals share the total.
    """

    SUPPORTED_RULE = "first_arrival_equal"
    SUPPORTED_QUALIFICATIONS = {
        "all_detected_events",
        "terminal_success_chain",
    }
    SUPPORTED_TRACE_RULES = {
        "endpoint_arrival_only",
        "causal_prefix_decay",
    }

    def __init__(
        self,
        credit_per_event: float = 1.0,
        event_classes: Sequence[str] = ("E3",),
        assignment_rule: str = SUPPORTED_RULE,
        qualification: str = "all_detected_events",
        trace_rule: str = "endpoint_arrival_only",
        trace_decay: float = 0.8,
    ) -> None:
        if assignment_rule != self.SUPPORTED_RULE:
            raise ValueError(f"Unsupported fixed event-alignment rule: {assignment_rule}")
        if not np.isfinite(float(credit_per_event)) or float(credit_per_event) <= 0.0:
            raise ValueError("credit_per_event must be finite and positive.")
        classes = tuple(str(value) for value in event_classes)
        if not classes or any(value not in {"E1", "E2", "E3"} for value in classes):
            raise ValueError("event_classes must be a non-empty subset of E1/E2/E3.")
        qualification = str(qualification)
        if qualification not in self.SUPPORTED_QUALIFICATIONS:
            raise ValueError(f"Unsupported event qualification: {qualification}")
        trace_rule = str(trace_rule)
        if trace_rule not in self.SUPPORTED_TRACE_RULES:
            raise ValueError(f"Unsupported event trace rule: {trace_rule}")
        if not np.isfinite(float(trace_decay)) or not 0.0 < float(trace_decay) <= 1.0:
            raise ValueError("trace_decay must be finite and in (0, 1].")
        self.credit_per_event = float(credit_per_event)
        self.event_classes = classes
        self.assignment_rule = assignment_rule
        self.qualification = qualification
        self.trace_rule = trace_rule
        self.trace_decay = float(trace_decay)

    @staticmethod
    def _first_arrival_step(
        factual_cache: Sequence[Any],
        transitions: Sequence[Any],
        robot: int,
        region_index: int,
        event_index: int,
    ) -> int | None:
        for index in range(event_index + 1):
            before = factual_cache[index].state
            after = transitions[index].state
            if (
                not bool(before.O[robot, region_index])
                and bool(after.O[robot, region_index])
            ):
                return index
        return None

    def align_episode(
        self,
        factual_cache: Sequence[Any],
        transitions: Sequence[Any],
    ) -> Tuple[EventCreditSample, ...]:
        cache = tuple(factual_cache)
        results = tuple(transitions)
        if len(cache) != len(results):
            raise ValueError("Event alignment requires one pre-action record per transition.")
        if self.qualification == "terminal_success_chain" and not (
            results and bool(results[-1].done) and bool(results[-1].success)
        ):
            return ()
        assignments: list[EventCreditSample] = []
        for event_index, result in enumerate(results):
            if int(cache[event_index].absolute_time) != event_index:
                raise ValueError("Factual event history must be contiguous from time zero.")
            for event in result.events:
                if event.event_class not in self.event_classes or event.robot_pair is None:
                    continue
                if event.region_index is None:
                    raise ValueError("A closure event requires a region index.")
                region = int(event.region_index)
                arrivals: list[tuple[int, int]] = []
                for robot in event.robot_pair:
                    action_time = self._first_arrival_step(
                        cache, results, int(robot), region, event_index
                    )
                    if action_time is not None:
                        arrivals.append((int(robot), action_time))
                if not arrivals:
                    continue
                credit = self.credit_per_event / float(len(arrivals))
                event_time = event_index + 1
                event_id = (
                    f"t{event_time}:o{int(event.order)}:"
                    f"r{event.robot_pair[0]}-{event.robot_pair[1]}:v{region}"
                )
                for robot, action_time in arrivals:
                    trace_start = 0 if self.trace_rule == "causal_prefix_decay" else action_time
                    for trace_time in range(trace_start, action_time + 1):
                        source = cache[trace_time]
                        if trace_time == action_time:
                            role = "completer" if action_time == event_index else "initiator"
                        else:
                            role = "enabling_trace"
                        assignments.append(
                            EventCreditSample(
                                observation=source.observations[robot],
                                action=source.factual_action[robot],
                                invalid_action_mask=source.action_masks[robot],
                                credit=credit * self.trace_decay ** (action_time - trace_time),
                                event_time=event_time,
                                action_time=trace_time,
                                robot=robot,
                                region_index=region,
                                event_class=event.event_class,
                                role=role,
                                event_id=event_id,
                            )
                        )
        return tuple(assignments)


class EventAlignedCreditModule:
    """Own fixed alignment, isolated replay, and bounded utility-margin loss."""

    def __init__(self, config: Mapping[str, Any], seed: int) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.loss_weight = float(config.get("loss_weight", 1.0))
        self.margin_scale = float(config.get("margin_scale", 1.0))
        self.batch_size = int(config.get("batch_size", 32))
        if self.loss_weight < 0.0 or not np.isfinite(self.loss_weight):
            raise ValueError("Event loss_weight must be finite and non-negative.")
        if self.margin_scale <= 0.0 or not np.isfinite(self.margin_scale):
            raise ValueError("Event margin_scale must be finite and positive.")
        if self.batch_size <= 0:
            raise ValueError("Event batch_size must be positive.")
        self.aligner = FixedRuleEventAligner(
            credit_per_event=float(config.get("credit_per_event", 1.0)),
            event_classes=config.get("event_classes", ("E3",)),
            assignment_rule=str(
                config.get("assignment_rule", FixedRuleEventAligner.SUPPORTED_RULE)
            ),
            qualification=str(config.get("qualification", "all_detected_events")),
            trace_rule=str(config.get("trace_rule", "endpoint_arrival_only")),
            trace_decay=float(config.get("trace_decay", 0.8)),
        )
        self.replay = EventCreditReplayBuffer(
            int(config.get("replay_capacity", 20000)), int(seed) + 89
        )
        self.assignment_count = 0
        self.observed_episode_count = 0
        self.qualified_episode_count = 0
        self.event_occurrence_count = 0
        self.event_ids: set[str] = set()
        self.role_counts: dict[str, int] = {"initiator": 0, "completer": 0}
        self.last_diagnostics: dict[str, Any] = {}

    def add_episode(
        self, factual_cache: Sequence[Any], transitions: Sequence[Any]
    ) -> Tuple[EventCreditSample, ...]:
        if not self.enabled:
            return ()
        self.observed_episode_count += 1
        samples = self.aligner.align_episode(factual_cache, transitions)
        if samples:
            self.qualified_episode_count += 1
            self.event_occurrence_count += len({sample.event_id for sample in samples})
        for sample in samples:
            self.replay.push(sample)
            self.assignment_count += 1
            self.event_ids.add(sample.event_id)
            self.role_counts[sample.role] = self.role_counts.get(sample.role, 0) + 1
        return samples

    def auxiliary_loss(
        self, network: torch.nn.Module, device: torch.device
    ) -> torch.Tensor | None:
        if not self.enabled or len(self.replay) == 0 or self.loss_weight == 0.0:
            self.last_diagnostics = {
                "sample_count": 0,
                "raw_loss": 0.0,
                "weighted_loss": 0.0,
            }
            return None
        samples = self.replay.sample(self.batch_size)
        observations = torch.as_tensor(
            np.stack([item.observation for item in samples]),
            dtype=torch.float32,
            device=device,
        )
        invalid = torch.as_tensor(
            np.stack([item.invalid_action_mask for item in samples]),
            dtype=torch.bool,
            device=device,
        )
        actions = torch.as_tensor(
            [item.action for item in samples], dtype=torch.long, device=device
        )
        credits = torch.as_tensor(
            [item.credit for item in samples], dtype=torch.float32, device=device
        )
        q_values = network(observations)
        row = torch.arange(q_values.shape[0], device=device)
        chosen = q_values[row, actions]
        alternatives = invalid.clone()
        alternatives[row, actions] = True
        has_alternative = (~alternatives).any(dim=1)
        if not bool(has_alternative.any()):
            self.last_diagnostics = {
                "sample_count": len(samples),
                "usable_count": 0,
                "raw_loss": 0.0,
                "weighted_loss": 0.0,
            }
            return None
        best_alternative = q_values.masked_fill(alternatives, -torch.inf).max(dim=1).values
        desired_margin = self.margin_scale * credits
        violations = torch.relu(
            desired_margin[has_alternative]
            - (chosen[has_alternative] - best_alternative[has_alternative])
        )
        raw_loss = violations.square().mean()
        weighted = self.loss_weight * raw_loss
        self.last_diagnostics = {
            "sample_count": len(samples),
            "usable_count": int(has_alternative.sum().detach().cpu().item()),
            "raw_loss": float(raw_loss.detach().cpu().item()),
            "weighted_loss": float(weighted.detach().cpu().item()),
            "mean_credit": float(credits.mean().detach().cpu().item()),
            "mean_delay": float(np.mean([item.delay for item in samples])),
            "max_delay": int(max(item.delay for item in samples)),
        }
        return weighted

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "assignment_rule": self.aligner.assignment_rule,
            "qualification": self.aligner.qualification,
            "trace_rule": self.aligner.trace_rule,
            "trace_decay": self.aligner.trace_decay,
            "event_classes": list(self.aligner.event_classes),
            "credit_per_event": self.aligner.credit_per_event,
            "replay_size": len(self.replay),
            "assignment_count": self.assignment_count,
            "observed_episode_count": self.observed_episode_count,
            "qualified_episode_count": self.qualified_episode_count,
            "event_occurrence_count": self.event_occurrence_count,
            "unique_event_count": len(self.event_ids),
            "role_counts": dict(self.role_counts),
            "last_loss": dict(self.last_diagnostics),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "replay": self.replay.state_dict(),
            "assignment_count": self.assignment_count,
            "observed_episode_count": self.observed_episode_count,
            "qualified_episode_count": self.qualified_episode_count,
            "event_occurrence_count": self.event_occurrence_count,
            "event_ids": sorted(self.event_ids),
            "role_counts": dict(self.role_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if bool(state["enabled"]) != self.enabled:
            raise ValueError("Checkpoint event-aligned enable flag differs from config.")
        self.replay.load_state_dict(state["replay"])
        self.assignment_count = int(state.get("assignment_count", 0))
        self.observed_episode_count = int(state.get("observed_episode_count", 0))
        self.qualified_episode_count = int(state.get("qualified_episode_count", 0))
        self.event_occurrence_count = int(state.get("event_occurrence_count", 0))
        self.event_ids = set(str(value) for value in state.get("event_ids", ()))
        self.role_counts = {
            str(key): int(value) for key, value in state.get("role_counts", {}).items()
        }


__all__ = [
    "EventAlignedCreditModule",
    "EventCreditReplayBuffer",
    "EventCreditSample",
    "FixedRuleEventAligner",
]
