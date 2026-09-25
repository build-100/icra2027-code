"""GEVD value-decomposition backbone with representative information and consistent branch replay.

There is one active Q-learning path: N=1 is the DQN regression and N>1 sums
shared local utilities once per joint transition.  MCBR is abstract-model data
augmentation, not a policy-gradient baseline or causal-credit estimator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from gevd.models.dqn import DQNAgent
from gevd.environments.prior_graph import PriorGraph
from gevd.training.event_credit import EventAlignedCreditModule
from gevd.training.learned_event_credit import LearnedEventCreditModule, make_event_credit
from gevd.environments.multi_robot import CentralState, GEVDMultiRobotEnv, StepResult, OBJECTIVE_VERSION
from gevd.models.networks import GraphUtilityNetwork
from gevd.training.replay import MixedReplaySampler, TransitionReplayBuffer
from gevd.training.policy_retention import SuccessfulWitnessRetention


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/simulation/seven_node/gevd.yaml"

DEFAULT_EVENT_ALIGNED_CONFIG = {
    "enabled": False,
    "mode": "fixed",
    "source": "factual_only",
    "event_classes": ["E3"],
    "assignment_rule": "first_arrival_equal",
    "qualification": "all_detected_events",
    "trace_rule": "endpoint_arrival_only",
    "trace_decay": 0.8,
    "credit_per_event": 1.0,
    "margin_scale": 1.0,
    "loss_weight": 0.25,
    "replay_capacity": 20000,
    "batch_size": 32,
}


def resolve_inside_project(path_value: Any) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must stay inside {PROJECT_ROOT}: {path}") from error
    return path


def _require_keys(section: Mapping[str, Any], names: Iterable[str], label: str) -> None:
    missing = [name for name in names if name not in section]
    if missing:
        raise ValueError(f"Missing {label} config key(s): {', '.join(missing)}")


def _integer(value: Any, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}.")
    return value


def _number(value: Any, name: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}." if minimum is not None else "."))
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("The GEVD config must be a mapping.")
    declared_objective = config.get("objective_version", OBJECTIVE_VERSION)
    if declared_objective != OBJECTIVE_VERSION:
        raise ValueError("Config declares a different scientific objective; use its isolated implementation.")
    for name in ("environment", "reward", "vdn", "mcbr", "guarantee", "output"):
        if name not in config or not isinstance(config[name], Mapping):
            raise ValueError(f"Missing config section: {name}")
    env, reward, vdn = config["environment"], config["reward"], config["vdn"]
    mcbr, guarantee, output = config["mcbr"], config["guarantee"], config["output"]
    _require_keys(env, ("graph_path", "map_width", "robot_count", "start_nodes", "t_max"), "environment")
    _require_keys(reward, ("alpha", "beta", "rho_v", "rho_g", "pair_factor_weight", "catalog_counting"), "reward")
    _require_keys(vdn, (
        "episodes", "hidden_dim", "gamma", "learning_rate", "batch_size",
        "factual_replay_capacity", "min_factual_replay", "epsilon_start",
        "epsilon_end", "epsilon_decay_steps", "target_update_interval",
        "updates_per_episode", "gradient_clip", "seed", "device",
    ), "vdn")
    _require_keys(mcbr, ("branch_replay_capacity", "branch_roots_per_episode", "branch_fraction"), "mcbr")
    _require_keys(guarantee, ("mode", "delta_s_bound", "successful_reference_distance", "strict_margin"), "guarantee")
    _require_keys(output, ("directory", "checkpoint_name", "summary_name", "routes_name"), "output")

    robot_count = _integer(env["robot_count"], "robot_count")
    if not isinstance(env["start_nodes"], (list, tuple)) or len(env["start_nodes"]) != robot_count:
        raise ValueError("start_nodes must contain exactly robot_count graph labels.")
    _integer(env["t_max"], "t_max")
    _number(env["map_width"], "map_width", 0.0)
    if reward["alpha"] != "reciprocal_num_regions" and _number(reward["alpha"], "alpha") <= 0.0:
        raise ValueError("alpha must be positive.")
    for name in ("beta", "rho_v", "rho_g"):
        _number(reward[name], name, 0.0)
    if reward["pair_factor_weight"] != "median_traversal" and _number(reward["pair_factor_weight"], "pair_factor_weight") <= 0.0:
        raise ValueError("pair_factor_weight must be positive.")
    if reward["catalog_counting"] != "robot_edge_and_pair_region_instances":
        raise ValueError("catalog_counting must be robot_edge_and_pair_region_instances.")

    _integer(vdn["episodes"], "episodes", True)
    _integer(vdn["hidden_dim"], "hidden_dim")
    if _number(vdn["gamma"], "gamma") != 1.0:
        raise ValueError("The paper-defined GEVD target requires gamma=1 exactly.")
    if _number(vdn["learning_rate"], "learning_rate") <= 0.0:
        raise ValueError("learning_rate must be positive.")
    batch = _integer(vdn["batch_size"], "batch_size")
    capacity = _integer(vdn["factual_replay_capacity"], "factual_replay_capacity")
    minimum = _integer(vdn["min_factual_replay"], "min_factual_replay")
    if not batch <= minimum <= capacity:
        raise ValueError("Require batch_size <= min_factual_replay <= factual capacity.")
    start, end = _number(vdn["epsilon_start"], "epsilon_start"), _number(vdn["epsilon_end"], "epsilon_end")
    if not 0.0 <= end <= start <= 1.0:
        raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
    _integer(vdn["epsilon_decay_steps"], "epsilon_decay_steps")
    _integer(vdn["target_update_interval"], "target_update_interval")
    _integer(vdn["updates_per_episode"], "updates_per_episode", True)
    if "updates_per_factual_step" in vdn:
        _number(
            vdn["updates_per_factual_step"],
            "updates_per_factual_step",
            0.0,
        )
    if vdn["gradient_clip"] is not None and _number(vdn["gradient_clip"], "gradient_clip") <= 0.0:
        raise ValueError("gradient_clip must be positive or null.")
    if isinstance(vdn["seed"], bool) or not isinstance(vdn["seed"], (int, np.integer)):
        raise TypeError("seed must be an integer.")
    if "cpu_threads" in vdn:
        _integer(vdn["cpu_threads"], "cpu_threads")
    if "graph_message_passing_layers" in vdn:
        layers = _integer(
            vdn["graph_message_passing_layers"], "graph_message_passing_layers"
        )
        if layers > 3:
            raise ValueError("graph_message_passing_layers is bounded to at most 3.")
    if "graph_residual_updates" in vdn and not isinstance(
        vdn["graph_residual_updates"], bool
    ):
        raise TypeError("graph_residual_updates must be boolean.")
    if "normalize_local_features" in vdn and not isinstance(
        vdn["normalize_local_features"], bool
    ):
        raise TypeError("normalize_local_features must be boolean.")
    if "td_reward_scale" in vdn and _number(
        vdn["td_reward_scale"], "td_reward_scale"
    ) <= 0.0:
        raise ValueError("td_reward_scale must be positive.")
    if vdn.get("network_initialization", "legacy_gain_0_1") not in {
        "legacy_gain_0_1", "relu_orthogonal"
    }:
        raise ValueError("Unsupported network_initialization.")

    event = config.get("event_aligned")
    if event is not None:
        if not isinstance(event, Mapping):
            raise TypeError("event_aligned must be a mapping when supplied.")
        resolved_event = copy.deepcopy(DEFAULT_EVENT_ALIGNED_CONFIG)
        resolved_event.update(dict(event))
        if resolved_event["mode"] not in {"fixed", "learned", "fixed_normalized", "paired_return"}:
            raise ValueError("Unsupported event_aligned.mode.")
        if not isinstance(resolved_event["enabled"], bool):
            raise TypeError("event_aligned.enabled must be boolean.")
        if resolved_event["source"] != "factual_only":
            raise ValueError("The fixed first version supports source: factual_only only.")
        if resolved_event["assignment_rule"] != "first_arrival_equal":
            raise ValueError("The fixed first version requires first_arrival_equal.")
        if resolved_event["qualification"] not in {
            "all_detected_events",
            "terminal_success_chain",
        }:
            raise ValueError(
                "event_aligned.qualification must be all_detected_events or "
                "terminal_success_chain."
            )
        if resolved_event["trace_rule"] not in {
            "endpoint_arrival_only",
            "causal_prefix_decay",
        }:
            raise ValueError(
                "event_aligned.trace_rule must be endpoint_arrival_only or "
                "causal_prefix_decay."
            )
        trace_decay = _number(resolved_event["trace_decay"], "event trace_decay")
        if not 0.0 < trace_decay <= 1.0:
            raise ValueError("event_aligned.trace_decay must be in (0, 1].")
        classes = resolved_event["event_classes"]
        if isinstance(classes, (str, bytes)) or not isinstance(classes, Sequence):
            raise TypeError("event_aligned.event_classes must be a sequence.")
        classes = [str(value) for value in classes]
        if not classes or len(set(classes)) != len(classes) or any(
            value not in {"E1", "E2", "E3"} for value in classes
        ):
            raise ValueError("event_aligned.event_classes must be a unique E1/E2/E3 subset.")
        if _number(resolved_event["credit_per_event"], "credit_per_event") <= 0.0:
            raise ValueError("event_aligned.credit_per_event must be positive.")
        if _number(resolved_event["margin_scale"], "margin_scale") <= 0.0:
            raise ValueError("event_aligned.margin_scale must be positive.")
        _number(resolved_event["loss_weight"], "event loss_weight", 0.0)
        _integer(resolved_event["replay_capacity"], "event replay_capacity")
        _integer(resolved_event["batch_size"], "event batch_size")

    _integer(mcbr["branch_replay_capacity"], "branch_replay_capacity")
    _integer(mcbr["branch_roots_per_episode"], "branch_roots_per_episode", True)
    eta = _number(mcbr["branch_fraction"], "branch_fraction")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("branch_fraction must be in [0, 1].")
    if guarantee["mode"] not in {"empirical", "certified_bound"}:
        raise ValueError("guarantee.mode must be empirical or certified_bound.")
    if _number(guarantee["strict_margin"], "strict_margin") <= 0.0:
        raise ValueError("strict_margin must be positive.")
    if guarantee["mode"] == "certified_bound":
        if guarantee["delta_s_bound"] is None or guarantee["successful_reference_distance"] is None:
            raise ValueError("certified_bound needs delta_s_bound and successful_reference_distance.")
        _number(guarantee["delta_s_bound"], "delta_s_bound", 0.0)
        _number(guarantee["successful_reference_distance"], "successful_reference_distance", 0.0)
    for name in ("directory", "checkpoint_name", "summary_name", "routes_name"):
        if not isinstance(output[name], str) or not output[name].strip():
            raise ValueError(f"output.{name} must be a non-empty path string.")


def load_config(path: Any = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = resolve_inside_project(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def resolve_device(requested: Any) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch_checkpoint(path: Any, map_location: Any = "cpu") -> Dict[str, Any]:
    from . import checkpoints
    try:
        return torch.load(path, map_location=map_location, weights_only=False, pickle_module=checkpoints)
    except TypeError:
        return torch.load(path, map_location=map_location, pickle_module=checkpoints)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _network_digest(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FactualPreAction:
    """A factual pre-action root source; branches never create this record."""

    state: CentralState
    factual_action: Tuple[int, ...]
    observations: np.ndarray
    action_masks: np.ndarray
    absolute_time: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", self.state.clone())
        observations = np.asarray(self.observations, dtype=np.float32).copy()
        masks = np.asarray(self.action_masks, dtype=np.bool_).copy()
        observations.setflags(write=False)
        masks.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "action_masks", masks)


class VDNTrainer:
    CHECKPOINT_VERSION = 4

    def __init__(self, env: GEVDMultiRobotEnv, config: Mapping[str, Any]) -> None:
        validate_config(config)
        if not isinstance(env, GEVDMultiRobotEnv):
            raise TypeError("env must be a GEVDMultiRobotEnv.")
        self.config = copy.deepcopy(dict(config))
        configured = self.config["environment"]
        if int(configured["robot_count"]) != env.num_robots:
            raise ValueError("Config robot_count does not match the central environment.")
        if tuple(configured["start_nodes"]) != env.start_labels:
            raise ValueError("Config start_nodes do not match the central environment.")
        if int(configured["t_max"]) != env.t_max:
            raise ValueError("Config t_max does not match the central environment.")
        self.env = env
        self.vdn_config, self.mcbr_config = self.config["vdn"], self.config["mcbr"]
        resolved_event = copy.deepcopy(DEFAULT_EVENT_ALIGNED_CONFIG)
        resolved_event.update(dict(self.config.get("event_aligned", {})))
        self.config["event_aligned"] = resolved_event
        self.event_config = self.config["event_aligned"]
        self.reward_config = self.config["reward"]
        self.seed = int(self.vdn_config["seed"])
        self.device = resolve_device(self.vdn_config["device"])
        if "cpu_threads" in self.vdn_config:
            torch.set_num_threads(int(self.vdn_config["cpu_threads"]))
        seed_everything(self.seed)
        self.guarantee_status = self._evaluate_guarantee()
        self.config["effective"] = {
            "algorithm": self.algorithm_name,
            "graph_and_factor_normalization": copy.deepcopy(env.derived_config),
            "environment_schema": copy.deepcopy(env.schema()),
            "guarantee": copy.deepcopy(self.guarantee_status),
            "implementation_choices": {
                "graph_encoder": {
                    "message_passing_layers": int(
                        self.vdn_config.get("graph_message_passing_layers", 1)
                    ),
                    "residual_updates": bool(
                        self.vdn_config.get("graph_residual_updates", False)
                    ),
                    "normalize_local_features": bool(
                        self.vdn_config.get("normalize_local_features", False)
                    ),
                    "normalization_scale": int(env.t_max),
                    "initialization": self.vdn_config.get(
                        "network_initialization", "legacy_gain_0_1"
                    ),
                    "centralized_metric_information": False,
                    "broadcast_visit_metadata": True,
                    "objective_version": OBJECTIVE_VERSION,
                },
                "td_reward_scale": float(
                    self.vdn_config.get("td_reward_scale", 1.0)
                ),
                "td_reward_scaling_changes_policy_objective": False,
                "pair_factor_weight": env.pair_factor_weight_spec,
                "factor_catalog_counting": "robot_edge_and_pair_region_instances",
                "mixed_batch_rounding": "deterministic_round_half_up",
                "event_aligned_credit": {
                    "relationship_to_vdn": "parallel_auxiliary_loss",
                    "changes_environment_reward": False,
                    "changes_q_tot_sum": False,
                    "source": self.event_config["source"],
                    "assignment_rule": self.event_config["assignment_rule"],
                    "qualification": self.event_config["qualification"],
                    "trace_rule": self.event_config["trace_rule"],
                    "trace_decay": float(self.event_config["trace_decay"]),
                    "mode": self.event_config["mode"],
                    "learned_credit_magnitude": self.event_config["mode"] == "learned",
                },
            },
        }
        self.agent = self._new_agent()
        self.factual_replay = TransitionReplayBuffer(int(self.vdn_config["factual_replay_capacity"]), self.seed + 11)
        self.branch_replay = TransitionReplayBuffer(int(self.mcbr_config["branch_replay_capacity"]), self.seed + 23)
        self.mixed_sampler = MixedReplaySampler(float(self.mcbr_config["branch_fraction"]), self.seed + 37)
        self.branch_rng = random.Random(self.seed + 53)
        self.branch_behavior_rng = random.Random(self.seed + 71)
        self.event_credit = make_event_credit(
            self.event_config, self.seed, self.env,
            reward_scale=float(self.vdn_config.get("td_reward_scale", 1.0)),
        )
        self.policy_retention = SuccessfulWitnessRetention(self.config.get("policy_retention"), self.seed)
        self.config["effective"]["implementation_choices"]["stability_revision"] = {
            "policy_retention": copy.deepcopy(self.policy_retention.config),
            "credit_updates_only_with_new_labels": bool(
                self.event_config.get("learned", {}).get("update_only_with_new_labels", False)
            ),
            "retention_role": "optional_self_generated_witness_regularizer_not_GCR_or_causal_credit",
        }
        self.output_directory = resolve_inside_project(self.config["output"]["directory"])
        self.episode_count = 0
        self.global_step = 0
        self.branch_transition_count = 0
        self.loss_history: List[float] = []
        self.event_loss_history: List[float] = []
        self.mix_history: List[Dict[str, int]] = []
        self.best_factual: Optional[Dict[str, Any]] = None
        self.last_factual_cache: Tuple[FactualPreAction, ...] = ()
        self.last_branch_traces: List[Dict[str, Any]] = []

    @property
    def algorithm_name(self) -> str:
        """Describe the backbone without mislabeling it as the complete GEVD method."""
        if self.event_config.get("enabled", False):
            return "GEVD backbone with optional event credit"
        return "GEVD w/o Retrospective Utility"

    def runtime_device_info(self) -> Dict[str, Any]:
        parameter_devices = sorted({str(value.device) for value in self.agent.online.parameters()})
        buffer_devices = sorted({str(value.device) for value in self.agent.online.buffers()})
        info = {
            "requested_device": str(self.vdn_config["device"]),
            "resolved_device": str(self.device),
            "online_parameter_devices": parameter_devices,
            "online_buffer_devices": buffer_devices,
            "last_gradient_device": getattr(self.agent, "last_gradient_device", None),
            "optimizer_updates": int(self.agent.update_steps),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cpu_threads": int(torch.get_num_threads()),
            "event_aligned": self.event_credit.diagnostics(),
        }
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            info.update({
                "cuda_device_index": int(index),
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_runtime": torch.version.cuda,
                "cuda_memory_allocated_mib": float(torch.cuda.memory_allocated(index) / 1024**2),
                "cuda_peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(index) / 1024**2),
            })
        return info

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VDNTrainer":
        validate_config(config)
        environment, reward = config["environment"], config["reward"]
        graph_path = resolve_inside_project(environment["graph_path"])
        start_nodes = tuple(environment["start_nodes"])
        prior = PriorGraph(str(graph_path), float(environment["map_width"]), start_nodes[0], resolve_device(config["vdn"]["device"]))
        env = GEVDMultiRobotEnv(
            prior, start_nodes, int(environment["t_max"]), reward["alpha"],
            float(reward["beta"]), float(reward["rho_v"]), float(reward["rho_g"]),
            reward["pair_factor_weight"],
        )
        return cls(env, config)

    def _new_agent(self) -> DQNAgent:
        spec = self.env.graph_tensor_spec(device="cpu")
        network = GraphUtilityNetwork(
            self.env.num_nodes, self.env.num_edges, self.env.action_dim,
            spec["edge_endpoints"], torch.stack((spec["edge_omega"], spec["edge_distance"]), dim=1),
            spec["neighbor_node_indices"], spec["neighbor_edge_indices"],
            int(self.vdn_config["hidden_dim"]),
            message_passing_layers=int(
                self.vdn_config.get("graph_message_passing_layers", 1)
            ),
            residual_updates=bool(
                self.vdn_config.get("graph_residual_updates", False)
            ),
            normalize_local_features=bool(
                self.vdn_config.get("normalize_local_features", False)
            ),
            t_max=self.env.t_max,
            initialization=self.vdn_config.get(
                "network_initialization", "legacy_gain_0_1"
            ),
            num_robots=self.env.num_robots,
        )
        return DQNAgent(
            self.env.observation_dim, self.env.action_dim, int(self.vdn_config["hidden_dim"]),
            float(self.vdn_config["learning_rate"]), 1.0,
            int(self.vdn_config["target_update_interval"]), self.vdn_config["gradient_clip"],
            self.seed, self.device, network=network, loss_kind="mse",
            td_reward_scale=float(self.vdn_config.get("td_reward_scale", 1.0)),
            double_dqn=self.vdn_config.get("double_dqn", True),
        )

    def _evaluate_guarantee(self) -> Dict[str, Any]:
        specification = self.config["guarantee"]
        if specification["mode"] == "empirical":
            return {"mode": "empirical", "condition_verified": False, "claim": "hyperparameters_only_no_Proposition_1_guarantee"}
        if specification.get("objective_version") != OBJECTIVE_VERSION:
            raise ValueError("A certified bound must explicitly apply to the representative objective; old S bounds cannot be reused.")
        delta_s = float(specification["delta_s_bound"])
        distance = float(specification["successful_reference_distance"])
        margin = float(specification["strict_margin"])
        bound = delta_s + float(self.reward_config["beta"]) * distance + margin
        verified = float(self.reward_config["rho_v"]) >= bound and float(self.reward_config["rho_g"]) >= bound
        if not verified:
            raise ValueError("certified_bound requested but rho_v/rho_g are below Lambda.")
        return {
            "mode": "certified_bound", "condition_verified": True, "delta_s_bound": delta_s,
            "successful_reference_distance": distance, "strict_margin": margin, "lambda": bound,
            "claim": "Proposition_1_sufficient_condition_verified_from_supplied_bound_and_witness",
        }

    def epsilon(self) -> float:
        start, end = float(self.vdn_config["epsilon_start"]), float(self.vdn_config["epsilon_end"])
        fraction = min(self.global_step / int(self.vdn_config["epsilon_decay_steps"]), 1.0)
        return float(start + fraction * (end - start))

    def _labels(self, positions: Sequence[int]) -> Tuple[Any, ...]:
        return tuple(self.env.prior.index_to_node[int(position)] for position in positions)

    @staticmethod
    def _terminal_masks(masks: np.ndarray) -> np.ndarray:
        return np.ones_like(masks, dtype=np.bool_)

    def _push(self, replay: TransitionReplayBuffer, observations: np.ndarray, actions: Sequence[int], result: StepResult, masks: np.ndarray) -> None:
        replay.push(
            observations, np.asarray(actions, dtype=np.int64), result.reward, result.observations,
            masks, self._terminal_masks(result.action_masks) if result.done else result.action_masks,
            result.done, reward_channels=result.reward_channels.as_dict(),
        )

    def _consider_best(self, episode: Mapping[str, Any]) -> None:
        if not episode["success"]:
            return
        candidate = {
            "success": True,
            "source": "best_factual",
            "routes": copy.deepcopy(episode["routes"]),
            "return": float(episode["return"]),
            "objective": float(episode["return"]),
            "joint_steps": int(episode["joint_steps"]),
            "total_distance": float(episode["total_distance"]),
            "final_structural_score": float(episode["final_structural_score"]),
            "information_diagnostics": copy.deepcopy(episode.get("information_diagnostics")),
        }
        if self.best_factual is None or candidate["objective"] > self.best_factual["objective"] + 1e-12:
            self.best_factual = candidate
        elif math.isclose(candidate["objective"], self.best_factual["objective"], abs_tol=1e-12):
            if repr(candidate["routes"]) < repr(self.best_factual["routes"]):
                self.best_factual = candidate

    def run_factual_episode(
        self,
        training: bool = True,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        if max_steps is not None and int(max_steps) < 1:
            raise ValueError("max_steps must be positive when supplied.")
        state, observations, masks = self.env.reset()
        initial_potential = self.env.potential(state)
        routes = [[label] for label in self.env.start_labels]
        cache: List[FactualPreAction] = []
        transitions: List[StepResult] = []
        actions_taken: List[Tuple[int, ...]] = []
        channels = {name: 0.0 for name in ("spectral", "coverage", "gauge", "travel")}
        total_reward = 0.0
        total_distance = 0.0
        while not self.env.done and (
            max_steps is None or len(transitions) < int(max_steps)
        ):
            actions = tuple(int(value) for value in self.agent.select_actions(
                observations, masks, epsilon=self.epsilon() if training else 0.0
            ))
            if training:
                cache.append(FactualPreAction(state, actions, observations, masks, state.time))
            result = self.env.step(actions)
            if training:
                self._push(self.factual_replay, observations, actions, result, masks)
                self.global_step += 1
            for robot, label in enumerate(self._labels(result.state.positions)):
                routes[robot].append(label)
            for name, value in result.reward_channels.as_dict().items():
                channels[name] += float(value)
            total_reward += result.reward
            total_distance += result.travel_distance
            transitions.append(result)
            actions_taken.append(actions)
            state, observations, masks = result.state, result.observations, result.action_masks

        final_potential = self.env.potential(state)
        residual = self.env.telescoping_residual(
            initial_potential, final_potential, transitions, float(self.reward_config["beta"])
        )
        if not math.isclose(residual, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"Episode reward failed to telescope: {residual!r}")
        if not math.isclose(total_reward, sum(channels.values()), abs_tol=1e-10):
            raise AssertionError("Four reward channels do not sum to the team return.")
        event_assignments = ()
        if training:
            if isinstance(self.event_credit, LearnedEventCreditModule):
                event_assignments = self.event_credit.add_episode(
                    cache, transitions, env=self.env, agent=self.agent
                )
            else:
                event_assignments = self.event_credit.add_episode(cache, transitions)
        episode = {
            "routes": routes,
            "actions": actions_taken,
            "success": bool(self.env.success),
            "return": float(total_reward),
            "joint_steps": int(state.time),
            "total_distance": float(total_distance),
            "reward_channels": channels,
            "initial_potential": float(initial_potential),
            "final_potential": float(final_potential),
            "final_structural_score": float(self.env.structural_score(state)),
            "information_diagnostics": self.env.information_diagnostics(state),
            "final_coverage_count": int(self.env.coverage_count(state)),
            "final_component_count": int(self.env.component_count(state)),
            "telescoping_residual": float(residual),
            "transitions": transitions,
            "pre_action_cache": tuple(cache),
            "terminal": bool(self.env.done),
            "budget_truncated": bool(not self.env.done),
            "event_credit_assignment_count": len(event_assignments),
            "event_credit_unique_events": len(
                {sample.event_id for sample in event_assignments}
            ),
            "event_credit_mean_delay": (
                float(np.mean([sample.delay for sample in event_assignments]))
                if event_assignments else None
            ),
        }
        self.validate_routes(routes)
        if training:
            self.episode_count += 1
            self.last_factual_cache = tuple(cache)
            self._consider_best(episode)
            self.policy_retention.consider(episode)
        return episode

    def enumerate_branch_roots(self, factual_cache: Optional[Sequence[FactualPreAction]] = None) -> List[Tuple[int, int, int]]:
        """Enumerate only factual roots in (time-index, robot, alternative) order."""

        cache = self.last_factual_cache if factual_cache is None else tuple(factual_cache)
        roots: List[Tuple[int, int, int]] = []
        for index, item in enumerate(cache):
            if item.absolute_time != item.state.time or item.absolute_time != index:
                raise ValueError("Factual cache must be contiguous and indexed by absolute time.")
            for robot in range(self.env.num_robots):
                for alternative in np.flatnonzero(~item.action_masks[robot]):
                    alternative = int(alternative)
                    if alternative != item.factual_action[robot]:
                        roots.append((index, robot, alternative))
        return roots

    def _sample_roots(self, cache: Sequence[FactualPreAction], maximum: Optional[int]) -> List[Tuple[int, int, int]]:
        roots = self.enumerate_branch_roots(cache)
        limit = int(self.mcbr_config["branch_roots_per_episode"] if maximum is None else maximum)
        if limit < 0:
            raise ValueError("maximum branch roots cannot be negative.")
        return self.branch_rng.sample(roots, min(limit, len(roots))) if roots and limit else []

    def _run_branch(
        self,
        cache: Sequence[FactualPreAction],
        root: Tuple[int, int, int],
        frozen_network: torch.nn.Module,
        frozen_epsilon: float,
        digest: str,
    ) -> Dict[str, Any]:
        root_index, changed_robot, alternative = root
        factual = cache[root_index]
        observations, masks = self.env.restore(factual.state)
        if not np.array_equal(observations, factual.observations) or not np.array_equal(masks, factual.action_masks):
            raise AssertionError("Restored branch root differs from the factual pre-action state.")
        root_actions = list(factual.factual_action)
        root_actions[changed_robot] = int(alternative)
        changed = [robot for robot in range(self.env.num_robots) if root_actions[robot] != factual.factual_action[robot]]
        if changed != [changed_robot]:
            raise AssertionError("A root must replace exactly one robot action.")

        records: List[Dict[str, Any]] = []
        first = True
        while not self.env.done:
            before = self.env.snapshot()
            if first:
                actions = tuple(root_actions)
                first = False
            else:
                actions = tuple(int(value) for value in self.agent.select_actions(
                    observations, masks, epsilon=frozen_epsilon,
                    network=frozen_network, rng=self.branch_behavior_rng,
                ))
            result = self.env.step(actions)
            self._push(self.branch_replay, observations, actions, result, masks)
            self.branch_transition_count += 1
            records.append({
                "before_state": before,
                "after_state": result.state,
                "actions": actions,
                "reward": result.reward,
                "reward_channels": result.reward_channels.as_dict(),
                "action_masks": np.asarray(masks, dtype=np.bool_).copy(),
                "next_action_masks": self._terminal_masks(result.action_masks) if result.done else np.asarray(result.action_masks).copy(),
                "done": result.done,
                "success": result.success,
                "events": result.events,
            })
            observations, masks = result.observations, result.action_masks
            if result.done:
                break
        if _network_digest(frozen_network) != digest:
            raise AssertionError("Frozen branch behavior parameters changed inside the branch.")
        return {
            "root": root,
            "changed_robot": changed_robot,
            "factual_root_action": factual.factual_action,
            "branch_root_action": tuple(root_actions),
            "absolute_start_time": factual.absolute_time,
            "absolute_end_time": self.env.state.time,
            "frozen_epsilon": float(frozen_epsilon),
            "frozen_network_digest": digest,
            "records": records,
            "success": bool(self.env.success),
        }

    def generate_branches(
        self,
        factual_cache: Optional[Sequence[FactualPreAction]] = None,
        maximum: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cache = self.last_factual_cache if factual_cache is None else tuple(factual_cache)
        roots = self._sample_roots(cache, maximum)
        if not roots:
            self.last_branch_traces = []
            return []
        frozen_network = self.agent.frozen_network()
        frozen_epsilon = self.epsilon()
        digest = _network_digest(frozen_network)
        traces = [self._run_branch(cache, root, frozen_network, frozen_epsilon, digest) for root in roots]
        if len({trace["root"] for trace in traces}) != len(traces):
            raise AssertionError("Branch roots were not sampled without replacement.")
        self.last_branch_traces = traces
        return traces

    def _updates_if_ready(self, factual_steps: Optional[int] = None) -> List[float]:
        if len(self.factual_replay) < int(self.vdn_config["min_factual_replay"]):
            return []
        if "updates_per_factual_step" in self.vdn_config:
            if factual_steps is None:
                raise ValueError(
                    "factual_steps is required when updates_per_factual_step is configured."
                )
            update_count = int(
                round(
                    float(self.vdn_config["updates_per_factual_step"])
                    * int(factual_steps)
                )
            )
        else:
            update_count = int(self.vdn_config["updates_per_episode"])
        return self._learn_updates(update_count)

    def _learn_updates(self, update_count: int) -> List[float]:
        """Apply an explicit quota; scheduling is separate from the TD update."""
        losses = []
        for _ in range(update_count):
            batch, counts = self.mixed_sampler.sample(
                self.factual_replay, self.branch_replay, int(self.vdn_config["batch_size"])
            )
            event_loss = self.event_credit.auxiliary_loss(
                self.agent.online, self.device
            )
            retention_loss = self.policy_retention.auxiliary_loss(self.agent.online, self.device)
            auxiliary_loss = event_loss
            if retention_loss is not None:
                auxiliary_loss = retention_loss if event_loss is None else event_loss + retention_loss
            loss = self.agent.learn(batch, auxiliary_loss=auxiliary_loss)
            self.agent.diagnostic_history[-1].update(
                event_credit_loss=float(event_loss.detach()) if event_loss is not None else 0.0,
                event_aligned_auxiliary_loss=float(event_loss.detach()) if event_loss is not None else 0.0,
                combined_auxiliary_loss=self.agent.last_auxiliary_loss,
                policy_retention_loss=self.policy_retention.last_loss,
            )
            if not math.isfinite(loss):
                raise FloatingPointError("GEVD team TD loss became NaN or Inf.")
            self.loss_history.append(float(loss))
            self.event_loss_history.append(float(event_loss.detach()) if event_loss is not None else 0.0)
            self.mix_history.append(counts)
            losses.append(float(loss))
        return losses

    def train(self, episodes: Optional[int] = None) -> List[Dict[str, Any]]:
        count = int(self.vdn_config["episodes"] if episodes is None else episodes)
        if count < 0:
            raise ValueError("episodes cannot be negative.")
        self.vdn_config["episodes"] = count
        summaries = []
        for _ in range(count):
            factual = self.run_factual_episode(training=True)
            traces = self.generate_branches(factual["pre_action_cache"])
            losses = self._updates_if_ready(factual["joint_steps"])
            summaries.append({
                "success": factual["success"], "return": factual["return"],
                "joint_steps": factual["joint_steps"], "total_distance": factual["total_distance"],
                "reward_channels": factual["reward_channels"],
                "final_coverage_count": factual["final_coverage_count"],
                "final_component_count": factual["final_component_count"],
                "telescoping_residual": factual["telescoping_residual"],
                "branch_roots": [trace["root"] for trace in traces],
                "branch_transitions": sum(len(trace["records"]) for trace in traces),
                "successful_branches": sum(bool(trace["success"]) for trace in traces),
                "successful_branch_transitions": sum(
                    len(trace["records"]) for trace in traces if trace["success"]
                ),
                "event_credit_assignment_count": factual["event_credit_assignment_count"],
                "event_credit_unique_events": factual["event_credit_unique_events"],
                "event_credit_mean_delay": factual["event_credit_mean_delay"],
                "losses": losses,
            })
        return summaries

    def train_to_factual_budget(
        self,
        factual_transitions: int,
        progress_callback: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Train to an exact factual-transition budget.

        Complete factual episodes retain normal MCBR generation.  If the final
        budget boundary falls inside an episode, the nonterminal prefix is
        retained as ordinary one-step replay (its last transition still
        bootstraps) but is deliberately not used as an MCBR episode root set.
        This preserves the absorbing task semantics while making learning-run
        budgets exactly comparable.
        """

        budget = int(factual_transitions)
        if budget < 1:
            raise ValueError("factual_transitions must be positive.")
        if self.global_step > budget:
            raise ValueError("The requested budget is below the current checkpoint step.")
        summaries = []
        while self.global_step < budget:
            remaining = budget - self.global_step
            factual = self.run_factual_episode(training=True, max_steps=remaining)
            traces = (
                self.generate_branches(factual["pre_action_cache"])
                if factual["terminal"]
                else []
            )
            losses = self._updates_if_ready(factual["joint_steps"])
            summary = {
                "success": factual["success"],
                "return": factual["return"],
                "joint_steps": factual["joint_steps"],
                "total_distance": factual["total_distance"],
                "reward_channels": factual["reward_channels"],
                "final_coverage_count": factual["final_coverage_count"],
                "final_component_count": factual["final_component_count"],
                "telescoping_residual": factual["telescoping_residual"],
                "terminal": factual["terminal"],
                "budget_truncated": factual["budget_truncated"],
                "branch_roots": [trace["root"] for trace in traces],
                "branch_transitions": sum(len(trace["records"]) for trace in traces),
                "successful_branches": sum(bool(trace["success"]) for trace in traces),
                "successful_branch_transitions": sum(
                    len(trace["records"]) for trace in traces if trace["success"]
                ),
                "event_credit_assignment_count": factual["event_credit_assignment_count"],
                "event_credit_unique_events": factual["event_credit_unique_events"],
                "event_credit_mean_delay": factual["event_credit_mean_delay"],
                "losses": losses,
            }
            summaries.append(summary)
            if progress_callback is not None:
                progress_callback(self, copy.deepcopy(summary))
        if self.global_step != budget:
            raise AssertionError("Factual-transition budget was not met exactly.")
        return summaries

    def evaluate(self) -> Dict[str, Any]:
        episode = self.run_factual_episode(training=False)
        return {
            "success": episode["success"], "source": "decentralized_greedy",
            "routes": episode["routes"], "return": episode["return"],
            "objective": episode["return"], "joint_steps": episode["joint_steps"],
            "total_distance": episode["total_distance"],
            "final_structural_score": episode["final_structural_score"],
            "information_diagnostics": copy.deepcopy(episode.get("information_diagnostics")),
            "reward_channels": episode["reward_channels"],
            "telescoping_residual": episode["telescoping_residual"],
        }

    def select_final_plan(self, greedy: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        greedy = dict(self.evaluate() if greedy is None else greedy)
        candidates = ([greedy] if greedy.get("success") else []) + (
            [copy.deepcopy(self.best_factual)] if self.best_factual is not None else []
        )
        if not candidates:
            return {
                "success": False, "source": "none", "routes": None,
                "reason": "Neither decentralized greedy nor any factual rollout succeeded.",
            }
        winner = max(candidates, key=lambda value: (float(value["objective"]), value.get("source") == "decentralized_greedy"))
        self.validate_routes(winner["routes"])
        return winner

    def validate_routes(self, routes: Sequence[Sequence[Any]]) -> bool:
        if len(routes) != self.env.num_robots:
            raise ValueError("There must be exactly one route per robot.")
        lengths = {len(route) for route in routes}
        if len(lengths) != 1:
            raise ValueError("Synchronous no-stay routes must have equal lengths.")
        length = next(iter(lengths))
        if length < 1 or length > self.env.t_max + 1:
            raise ValueError("A route must contain between 1 and T_max+1 nodes.")
        for robot, route in enumerate(routes):
            if route[0] != self.env.start_labels[robot]:
                raise ValueError(f"Robot {robot} route starts at the wrong node.")
            for current, following in zip(route, route[1:]):
                if current == following:
                    raise ValueError("Stay/wait is not part of the model.")
                self.env.prior.action_for_neighbor(current, following)
        return True

    def checkpoint_path(self) -> Path:
        return resolve_inside_project(self.output_directory / self.config["output"]["checkpoint_name"])

    def summary_path(self) -> Path:
        return resolve_inside_project(self.output_directory / self.config["output"]["summary_name"])

    def routes_path(self) -> Path:
        return resolve_inside_project(self.output_directory / self.config["output"]["routes_name"])

    def save_checkpoint(self, path: Any = None) -> Path:
        destination = self.checkpoint_path() if path is None else resolve_inside_project(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": self.CHECKPOINT_VERSION,
            "algorithm": self.algorithm_name,
            "agent": self.agent.state_dict(),
            "factual_replay": self.factual_replay.state_dict(),
            "branch_replay": self.branch_replay.state_dict(),
            "mixed_sampler": self.mixed_sampler.state_dict(),
            "effective_config": copy.deepcopy(self.config),
            "environment_schema": copy.deepcopy(self.env.schema()),
            "episode_count": self.episode_count,
            "global_step": self.global_step,
            "branch_transition_count": self.branch_transition_count,
            "loss_history": list(self.loss_history),
            "mix_history": copy.deepcopy(self.mix_history),
            "event_aligned": self.event_credit.state_dict(),
            "policy_retention": self.policy_retention.state_dict(),
            "event_loss_history": list(self.event_loss_history),
            "best_factual": copy.deepcopy(self.best_factual),
            "seed": self.seed,
            "branch_rng_state": self.branch_rng.getstate(),
            "branch_behavior_rng_state": self.branch_behavior_rng.getstate(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, destination)
        return destination

    def load_checkpoint(self, path: Any) -> None:
        payload = load_torch_checkpoint(resolve_inside_project(path), map_location="cpu")
        version = int(payload.get("format_version", -1))
        if version != self.CHECKPOINT_VERSION:
            raise ValueError("Legacy checkpoint/replay is incompatible with first-visit owners, broadcast observations and representative rewards. Use the isolated old code for old checkpoints; new training must start fresh.")
        # Historical names are accepted only for reading existing checkpoint metadata.
        if payload.get("algorithm") not in {
            "GEVD",
            "GEVD w/o Retrospective Utility",
            "GEVD backbone with optional event credit",
            "GEVD w/o Collective Value Decomposition",
            "GEVD w/o Explicit Gauge-Nullity Term",
            "GEVD w/o Representative-Pose Marginalization",
            "B w/o Representative-Pose Marginalization",
            "Full GEVD",
            "GEDV",
            "w/ Paired Search",
            "GEVD-N",
            "GAUGE-VDN-MCBR",
            "GAUGE-VDN-MCBR-EVENT-ALIGNED",
            "GAUGE-VDN-MCBR-WITHOUT-EVENT-CREDIT",
            "GAUGE-LEARNED-EVENT-CREDIT",
            "GAUGE-NORMALIZED-FIXED-EVENT-CREDIT",
        }:
            raise ValueError("Unsupported or mismatched GEVD checkpoint.")
        if payload["environment_schema"] != self.env.schema():
            raise ValueError("Checkpoint graph/factor/robot/action schema differs.")
        restored = copy.deepcopy(payload["effective_config"])
        if self.config.get("full_gevd_ablation") != restored.get("full_gevd_ablation"):
            raise ValueError("GEVD ablation differs from checkpoint; start a fresh run.")
        if self.config.get("training_structure") != restored.get("training_structure"):
            raise ValueError("Training structure differs from checkpoint; start a fresh run.")
        if self.config.get("retrospective_utility") != restored.get("retrospective_utility"):
            raise ValueError("Retrospective utility configuration differs from checkpoint; start a fresh run.")
        # Paths and the runtime device belong to this checkout. The graph schema
        # was compared above; all learned state and scientific settings remain saved values.
        restored["environment"]["graph_path"] = self.config["environment"]["graph_path"]
        restored["output"]["directory"] = self.config["output"]["directory"]
        restored["vdn"]["device"] = self.config["vdn"]["device"]
        validate_config(restored)
        resolve_inside_project(restored["environment"]["graph_path"])
        if int(restored["vdn"]["seed"]) != int(payload["seed"]):
            raise ValueError("Checkpoint seed conflicts with its effective config.")
        if restored["effective"]["graph_and_factor_normalization"] != self.env.derived_config:
            raise ValueError("Checkpoint omega normalization differs from the environment.")

        expected_event = self.event_config
        incoming_event = restored.get("event_aligned", {})
        if "paired_return" in (expected_event.get("mode"), incoming_event.get("mode")):
            expected_signature = (expected_event.get("mode"), bool(expected_event.get("enabled", False)))
            incoming_signature = (incoming_event.get("mode"), bool(incoming_event.get("enabled", False)))
            if expected_signature != incoming_signature:
                raise ValueError("Paired-return credit mode differs from checkpoint; old credit checkpoints cannot resume the new model.")

        self.config = restored
        self.vdn_config, self.mcbr_config = restored["vdn"], restored["mcbr"]
        resolved_event = copy.deepcopy(DEFAULT_EVENT_ALIGNED_CONFIG)
        resolved_event.update(dict(restored.get("event_aligned", {})))
        restored["event_aligned"] = resolved_event
        self.event_config = restored["event_aligned"]
        self.reward_config = restored["reward"]
        self.seed = int(payload["seed"])
        self.device = resolve_device(self.vdn_config["device"])
        self.guarantee_status = copy.deepcopy(restored["effective"]["guarantee"])
        self.output_directory = resolve_inside_project(restored["output"]["directory"])
        self.agent = self._new_agent()
        self.agent.load_state_dict(payload["agent"])
        self.factual_replay = TransitionReplayBuffer(int(self.vdn_config["factual_replay_capacity"]), self.seed + 11)
        self.branch_replay = TransitionReplayBuffer(int(self.mcbr_config["branch_replay_capacity"]), self.seed + 23)
        self.factual_replay.load_state_dict(payload["factual_replay"])
        self.branch_replay.load_state_dict(payload["branch_replay"])
        self.mixed_sampler = MixedReplaySampler(float(self.mcbr_config["branch_fraction"]), self.seed + 37)
        self.mixed_sampler.load_state_dict(payload["mixed_sampler"])
        self.branch_rng = random.Random(self.seed + 53)
        self.branch_rng.setstate(payload["branch_rng_state"])
        self.branch_behavior_rng = random.Random(self.seed + 71)
        self.branch_behavior_rng.setstate(payload["branch_behavior_rng_state"])
        self.event_credit = make_event_credit(
            self.event_config, self.seed, self.env,
            reward_scale=float(self.vdn_config.get("td_reward_scale", 1.0)),
        )
        self.policy_retention = SuccessfulWitnessRetention(self.config.get("policy_retention"), self.seed)
        if "policy_retention" in payload:
            self.policy_retention.load_state_dict(payload["policy_retention"])
        elif self.policy_retention.enabled:
            raise ValueError("Enabled retention checkpoint is missing its witness state")
        if version == self.CHECKPOINT_VERSION:
            self.event_credit.load_state_dict(payload["event_aligned"])
            if getattr(self.event_credit, "mode", None) == "paired_return":
                self.event_credit._agent = self.agent
        elif self.event_credit.enabled:
            raise ValueError("A legacy checkpoint cannot resume an event-aligned run.")
        self.episode_count = int(payload["episode_count"])
        self.global_step = int(payload["global_step"])
        self.branch_transition_count = int(payload["branch_transition_count"])
        self.loss_history = [float(value) for value in payload["loss_history"]]
        self.mix_history = copy.deepcopy(payload["mix_history"])
        self.event_loss_history = [
            float(value) for value in payload.get("event_loss_history", [])
        ]
        self.best_factual = copy.deepcopy(payload["best_factual"])
        self.last_factual_cache, self.last_branch_traces = (), []
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state"])

    def write_json(self, value: Mapping[str, Any], path: Any) -> Path:
        destination = resolve_inside_project(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2)
        return destination

    def write_summary(self, summary: Mapping[str, Any], path: Any = None) -> Path:
        return self.write_json(summary, self.summary_path() if path is None else path)

    def write_routes(self, plan: Mapping[str, Any], path: Any = None) -> Path:
        return self.write_json(plan, self.routes_path() if path is None else path)


# Compatibility import only; this is an alias to the one active learner/trainer.
DQNTrainer = VDNTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the GEVD backbone.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--factual-budget", type=int,
                        help="Explicit factual transition budget for a fresh representative-objective run.")
    parser.add_argument("--search-budget", type=int,
                        help="Total simulator-call ceiling, including branches, credit and candidate verification.")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.search_budget is not None:
        if args.resume or args.evaluate_only or args.factual_budget is not None or args.episodes is not None:
            parser.error("Search-budget execution requires a fresh run and cannot combine other budget/evaluation modes.")
        from gevd.training.search import run_budgeted_config
        print(json.dumps(_jsonable(run_budgeted_config(config,args.search_budget)),ensure_ascii=False,indent=2))
        return
    if config.get('search',{}).get('verified_complete_candidates',False):
        parser.error("The GEVD backbone uses verified search candidates; supply --search-budget. Historical modes require their frozen historical configuration.")
    if args.episodes is not None and args.factual_budget is not None:
        parser.error("Choose an episode budget or a factual transition budget, not both.")
    trainer = VDNTrainer.from_config(config)
    print(json.dumps({"runtime_device_start": trainer.runtime_device_info()}, ensure_ascii=False))
    if args.resume:
        trainer.load_checkpoint(args.resume)
    if args.evaluate_only:
        print(json.dumps(_jsonable(trainer.select_final_plan(trainer.evaluate())), ensure_ascii=False, indent=2))
        return
    if args.factual_budget is None and args.episodes is None and int(trainer.vdn_config["episodes"]) == 0:
        parser.error("This configuration has no implicit training budget; supply --factual-budget or --episodes.")
    training = (trainer.train_to_factual_budget(args.factual_budget)
                if args.factual_budget is not None else trainer.train(args.episodes))
    checkpoint = trainer.save_checkpoint()
    plan = trainer.select_final_plan(trainer.evaluate())
    routes_path = trainer.write_routes(plan)
    summary = {
        "algorithm": trainer.algorithm_name,
        "episodes_this_run": len(training),
        "episode_count": trainer.episode_count,
        "factual_steps": trainer.global_step,
        "updates": trainer.agent.update_steps,
        "factual_replay_size": len(trainer.factual_replay),
        "branch_replay_size": len(trainer.branch_replay),
        "branch_transitions": trainer.branch_transition_count,
        "event_aligned": trainer.event_credit.diagnostics(),
        "losses_finite": bool(all(math.isfinite(loss) for loss in trainer.loss_history)),
        "runtime_device": trainer.runtime_device_info(),
        "guarantee": trainer.guarantee_status,
        "final_plan": plan,
        "checkpoint": str(checkpoint),
        "routes": str(routes_path),
    }
    summary_path = trainer.write_summary(summary)
    summary["summary"] = str(summary_path)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
