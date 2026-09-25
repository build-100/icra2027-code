"""Deterministic synchronous multi-robot abstract environment for GEVD.

The XML loader and the canonical padded neighbor/action mapping remain owned by
``PriorGraph``.  This module adds one central simulator whose complete dynamic
state is ``(M, O, owners, positions, time)``.  In particular, the pose graph, closures,
gauge potential, and rewards contain no hidden mutable history and can be
reconstructed exactly after a replay/branch restore.

``True`` always denotes an invalid padded action in masks returned here, as in
the first-stage DQN implementation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

from .prior_graph import PriorGraph


PoseNode = Tuple[int, int]  # (robot index, dense region index)
OBJECTIVE_VERSION = "first_visit_representative_schur_star_unique_motion_v1"
OBSERVATION_VERSION = "self_history_team_visitors_representatives_v1"


def _readonly_array(value: Any, dtype: np.dtype, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}.")
    array = np.array(array, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class CentralState:
    """Immutable, exactly restorable centralized simulator state.

    ``positions`` contains dense node indices, not external graph labels.  The
    environment validates shapes and semantic invariants on reset/restore/step.
    """

    M: np.ndarray
    O: np.ndarray
    positions: np.ndarray
    time: int
    owners: np.ndarray  # first team visitor per region; -1 means uncovered

    def __post_init__(self) -> None:
        raw_M = np.asarray(self.M)
        raw_O = np.asarray(self.O)
        raw_positions = np.asarray(self.positions)
        if np.asarray(self.owners).dtype.kind not in "iu":
            raise TypeError("owners must contain integer robot indices or -1.")
        if raw_M.dtype.kind not in "iu" or raw_M.dtype.kind == "b":
            raise TypeError("M must contain integer traversal counts.")
        if raw_O.dtype.kind != "b":
            raise TypeError("O must contain boolean visited bits.")
        if raw_positions.dtype.kind not in "iu" or raw_positions.dtype.kind == "b":
            raise TypeError("positions must contain integer dense node indices.")
        object.__setattr__(self, "M", _readonly_array(self.M, np.int64, 2, "M"))
        object.__setattr__(self, "O", _readonly_array(self.O, np.bool_, 2, "O"))
        object.__setattr__(self, "owners", _readonly_array(self.owners, np.int64, 1, "owners"))
        object.__setattr__(
            self,
            "positions",
            _readonly_array(self.positions, np.int64, 1, "positions"),
        )
        if isinstance(self.time, (bool, np.bool_)) or not isinstance(
            self.time, (int, np.integer)
        ):
            raise TypeError("time must be an integer joint-step index.")
        if int(self.time) < 0:
            raise ValueError("time must be non-negative.")
        object.__setattr__(self, "time", int(self.time))

    def clone(self) -> "CentralState":
        return CentralState(self.M, self.O, self.positions, self.time, self.owners)

    def as_dict(self) -> dict:
        return {
            "M": np.array(self.M, copy=True),
            "O": np.array(self.O, copy=True),
            "positions": np.array(self.positions, copy=True),
            "time": self.time,
            "owners": np.array(self.owners, copy=True),
            "state_version": 2,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CentralState":
        if payload.get("state_version") != 2 or "owners" not in payload:
            raise ValueError("Legacy state lacks first-visit provenance; replay ordered routes instead.")
        return cls(
            M=payload["M"],
            O=payload["O"],
            positions=payload["positions"],
            time=payload["time"],
            owners=payload["owners"],
        )

    def equals(self, other: object) -> bool:
        return (
            isinstance(other, CentralState)
            and self.time == other.time
            and np.array_equal(self.M, other.M)
            and np.array_equal(self.O, other.O)
            and np.array_equal(self.owners, other.owners)
            and np.array_equal(self.positions, other.positions)
        )

    def __eq__(self, other: object) -> bool:
        return self.equals(other)


@dataclass(frozen=True)
class RewardChannels:
    spectral: float
    coverage: float
    gauge: float
    travel: float

    def __post_init__(self) -> None:
        for name in ("spectral", "coverage", "gauge", "travel"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite {name} reward channel: {value!r}.")
            object.__setattr__(self, name, value)

    @property
    def total(self) -> float:
        # This exact expression is the authoritative scalar team reward.
        return float(self.spectral + self.coverage + self.gauge + self.travel)

    def as_dict(self) -> dict:
        return {
            "spectral": self.spectral,
            "coverage": self.coverage,
            "gauge": self.gauge,
            "travel": self.travel,
        }


@dataclass(frozen=True)
class FactorEvent:
    """One newly instantiated factor in deterministic processing order."""

    order: int
    factor_type: str  # "motion" or "closure"
    event_class: str  # E1, E2, or E3
    endpoints: Tuple[PoseNode, PoseNode]
    normalized_weight: float
    raw_weight: float
    structural_delta_formula: float  # actual representative-score delta, NOT raw E1/E2/E3 formula
    raw_structural_delta_formula: float  # raw full-graph Matrix-Tree diagnostic only
    robot: Optional[int] = None
    edge_index: Optional[int] = None
    occurrence: Optional[int] = None
    robot_pair: Optional[Tuple[int, int]] = None
    region_index: Optional[int] = None


@dataclass(frozen=True)
class StepResult:
    """Atomic joint-step output."""

    state: CentralState
    observations: np.ndarray
    action_masks: np.ndarray
    actions: Tuple[int, ...]
    reward: float
    reward_channels: RewardChannels
    done: bool
    success: bool
    events: Tuple[FactorEvent, ...]
    structural_before: float
    structural_after: float
    potential_before: float
    potential_after: float
    components_before: int
    components_after: int
    travel_distance: float

    def __post_init__(self) -> None:
        observations = _readonly_array(
            self.observations, np.float32, 2, "observations"
        )
        masks = _readonly_array(self.action_masks, np.bool_, 2, "action_masks")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "action_masks", masks)
        if not math.isfinite(float(self.reward)):
            raise FloatingPointError("The scalar team reward must be finite.")
        if float(self.reward) != self.reward_channels.total:
            raise AssertionError("Scalar reward must be the exact sum of all four channels.")

    @property
    def next_state(self) -> CentralState:
        return self.state

    @property
    def next_observations(self) -> np.ndarray:
        return self.observations

    @property
    def next_action_masks(self) -> np.ndarray:
        return self.action_masks


def _numeric_positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _numeric_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _component_laplacian(
    graph: nx.MultiGraph, component: Iterable[PoseNode]
) -> Tuple[Tuple[PoseNode, ...], np.ndarray]:
    nodes = tuple(sorted(component))
    index = {node: position for position, node in enumerate(nodes)}
    laplacian = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    subgraph = graph.subgraph(nodes)
    for source, target, attributes in subgraph.edges(data=True):
        if source == target:
            raise ValueError("Pose-graph self factors are not admissible.")
        weight = _numeric_positive(attributes.get("weight"), "factor weight")
        i = index[source]
        j = index[target]
        laplacian[i, i] += weight
        laplacian[j, j] += weight
        laplacian[i, j] -= weight
        laplacian[j, i] -= weight
    return nodes, laplacian


def component_log_spanning_tree(
    graph: nx.MultiGraph,
    component: Iterable[PoseNode],
    anchor: Optional[PoseNode] = None,
) -> float:
    """Return componentwise ``log tau_omega`` via a reduced Laplacian.

    A singleton has spanning-tree value one and therefore log value zero.  No
    diagonal floor or eigenvalue threshold is used.
    """

    nodes, laplacian = _component_laplacian(graph, component)
    if not nodes:
        raise ValueError("A component cannot be empty.")
    if len(nodes) == 1:
        return 0.0
    if anchor is None:
        anchor = nodes[-1]
    if anchor not in nodes:
        raise ValueError(f"Anchor {anchor!r} is not in the component.")
    anchor_index = nodes.index(anchor)
    reduced = np.delete(np.delete(laplacian, anchor_index, axis=0), anchor_index, axis=1)
    sign, logdet = np.linalg.slogdet(reduced)
    if sign <= 0.0 or not math.isfinite(float(logdet)):
        raise FloatingPointError(
            "A connected positive-weight component produced a non-positive "
            "reduced-Laplacian determinant."
        )
    return float(logdet)


def effective_resistance(
    graph: nx.MultiGraph, source: PoseNode, target: PoseNode
) -> float:
    """Effective resistance inside one connected component using ``solve``."""

    if source == target:
        return 0.0
    if source not in graph or target not in graph:
        raise ValueError("Both effective-resistance endpoints must exist.")
    component = nx.node_connected_component(graph, source)
    if target not in component:
        raise ValueError("Effective resistance is only defined here within one component.")
    nodes, laplacian = _component_laplacian(graph, component)
    # Any local anchor is valid.  Using the last node is deterministic.
    anchor_index = len(nodes) - 1
    reduced = np.delete(np.delete(laplacian, anchor_index, axis=0), anchor_index, axis=1)
    injection = np.zeros(len(nodes), dtype=np.float64)
    injection[nodes.index(source)] = 1.0
    injection[nodes.index(target)] = -1.0
    reduced_injection = np.delete(injection, anchor_index)
    try:
        solution = np.linalg.solve(reduced, reduced_injection)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError(
            "The component reduced Laplacian is singular."
        ) from error
    resistance = float(reduced_injection @ solution)
    if not math.isfinite(resistance) or resistance < 0.0:
        raise FloatingPointError(f"Invalid effective resistance: {resistance!r}.")
    return resistance


def _spd_logdet(matrix: np.ndarray) -> float:
    """Strict SPD log determinant; no artificial prior or eigenvalue clipping."""
    if matrix.shape == (0, 0):
        return 0.0
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("Expected positive definite information block.") from error
    value = float(2.0 * np.log(np.diag(factor)).sum())
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite information determinant.")
    return value


def component_representative_information(
    graph: nx.MultiGraph,
    component: Iterable[PoseNode],
    representatives: Iterable[PoseNode],
    anchor: Optional[PoseNode] = None,
) -> dict:
    """Kron-reduce a floating scalar pose component to its output keyframes.

    This is exact for the weighted scalar Laplacian surrogate, not an SE(2)/SE(3)
    covariance model. One retained anchor removes only this component's gauge.
    """
    nodes, laplacian = _component_laplacian(graph, component)
    targets = set(representatives)
    retained = tuple(node for node in nodes if node in targets)
    auxiliary = tuple(node for node in nodes if node not in targets)
    if not retained:
        raise ValueError("Every completed pose component must contain an output representative.")
    q = [nodes.index(node) for node in retained]
    a = [nodes.index(node) for node in auxiliary]
    kron = laplacian[np.ix_(q, q)].copy()
    auxiliary_logdet = 0.0
    if a:
        block_aa = laplacian[np.ix_(a, a)]
        auxiliary_logdet = _spd_logdet(block_aa)
        block_aq = laplacian[np.ix_(a, q)]
        kron -= block_aq.T @ np.linalg.solve(block_aa, block_aq)
        kron = (kron + kron.T) * 0.5
    anchor = retained[-1] if anchor is None else anchor
    if anchor not in retained:
        raise ValueError("The scoring anchor must be a retained representative.")
    anchor_index = retained.index(anchor)
    reduced = np.delete(np.delete(kron, anchor_index, axis=0), anchor_index, axis=1)
    return {
        "retained": retained,
        "auxiliary": auxiliary,
        "anchor": anchor,
        "kron_laplacian": kron,
        "reduced_information": reduced,
        "representative_logdet": _spd_logdet(reduced),
        "auxiliary_conditional_logdet": auxiliary_logdet,
    }


class GEVDMultiRobotEnv:
    """Synchronous deterministic GEVD abstract simulator.

    Parameters use external graph labels only for ``start_nodes``.  Dynamic
    positions are stored as dense indices so non-contiguous labels remain safe.
    ``pair_factor_weight`` is either a positive scalar or
    ``"median_traversal"``.
    """

    def __init__(
        self,
        prior: PriorGraph,
        start_nodes: Sequence[Any],
        t_max: int,
        alpha: Any,
        beta: float,
        rho_V: float,
        rho_g: float,
        pair_factor_weight: Any = "median_traversal",
    ) -> None:
        if not isinstance(prior, PriorGraph):
            raise TypeError("prior must be a PriorGraph instance.")
        if isinstance(t_max, (bool, np.bool_)) or not isinstance(t_max, (int, np.integer)):
            raise TypeError("t_max must be an integer.")
        if int(t_max) < 1:
            raise ValueError("t_max must be at least one joint step.")
        if not start_nodes:
            raise ValueError("At least one robot start node is required.")

        self.prior = prior
        self.t_max = int(t_max)
        self.start_labels = tuple(start_nodes)
        missing = [label for label in self.start_labels if label not in prior.prior_graph]
        if missing:
            raise ValueError(f"Unknown robot start node(s): {missing!r}.")
        self.num_robots = len(self.start_labels)
        self.robot_indices = tuple(range(self.num_robots))
        self.num_nodes = prior.num_nodes
        self.node_labels = tuple(prior.node_labels)
        self.start_positions = np.asarray(
            [prior.node_to_index[label] for label in self.start_labels], dtype=np.int64
        )

        if nx.number_of_selfloops(prior.prior_graph):
            raise ValueError("Prior-graph self loops would create a forbidden stay action.")

        self.edge_endpoints = tuple(
            sorted(
                (
                    min(prior.node_to_index[source], prior.node_to_index[target]),
                    max(prior.node_to_index[source], prior.node_to_index[target]),
                )
                for source, target in prior.prior_graph.edges()
            )
        )
        self.num_edges = len(self.edge_endpoints)
        self.edge_to_index = {
            endpoints: index for index, endpoints in enumerate(self.edge_endpoints)
        }
        if len(self.edge_to_index) != self.num_edges:
            raise ValueError("The prior graph must not contain parallel planning edges.")

        edge_raw_weights = []
        edge_distances = []
        for edge_index, (source_index, target_index) in enumerate(self.edge_endpoints):
            source = prior.index_to_node[source_index]
            target = prior.index_to_node[target_index]
            attributes = prior.prior_graph.edges[source, target]
            raw_weight = attributes.get("omega", attributes.get("d_opt"))
            edge_raw_weights.append(
                _numeric_positive(raw_weight, f"omega/d_opt for edge {edge_index}")
            )
            distance = attributes.get("distance", attributes.get("weight"))
            edge_distances.append(
                _numeric_nonnegative(distance, f"distance for edge {edge_index}")
            )
        if not edge_raw_weights:
            raise ValueError("At least one admissible traversal edge is required (no stay action).")
        self.edge_raw_weights = np.asarray(edge_raw_weights, dtype=np.float64)
        self.edge_distances = np.asarray(edge_distances, dtype=np.float64)

        if pair_factor_weight == "median_traversal":
            self.pair_factor_weight_spec = "median_traversal"
            self.pair_raw_weight = float(np.median(self.edge_raw_weights))
        else:
            self.pair_factor_weight_spec = float(
                _numeric_positive(pair_factor_weight, "pair_factor_weight")
            )
            self.pair_raw_weight = self.pair_factor_weight_spec

        self.robot_pairs = tuple(combinations(self.robot_indices, 2))
        self.closure_catalog = tuple(
            (robot_a, robot_b, region_index)
            for robot_a, robot_b in self.robot_pairs
            for region_index in range(self.num_nodes)
        )
        self.closure_to_index = {
            closure: index for index, closure in enumerate(self.closure_catalog)
        }

        # Catalog multiplicity is scientific: every (robot, admissible edge) and
        # every (unordered robot pair, region) is one catalog instance.
        catalog_weights = np.concatenate(
            [
                np.tile(self.edge_raw_weights, self.num_robots),
                np.full(len(self.closure_catalog), self.pair_raw_weight, dtype=np.float64),
            ]
        )
        self.omega_ref = _numeric_positive(float(np.median(catalog_weights)), "omega_ref")
        self.edge_normalized_weights = self.edge_raw_weights / self.omega_ref
        self.pair_normalized_weight = self.pair_raw_weight / self.omega_ref

        if alpha == "reciprocal_num_regions":
            self.alpha_spec = "reciprocal_num_regions"
            self.alpha = 1.0 / float(self.num_nodes)
        else:
            self.alpha_spec = float(_numeric_positive(alpha, "alpha"))
            self.alpha = self.alpha_spec
        self.beta = _numeric_nonnegative(beta, "beta")
        self.rho_V = _numeric_nonnegative(rho_V, "rho_V")
        self.rho_g = _numeric_nonnegative(rho_g, "rho_g")

        self.neighbor_edge_indices = np.full(
            (self.num_nodes, prior.action_dim), -1, dtype=np.int64
        )
        self.neighbor_node_indices = np.full(
            (self.num_nodes, prior.action_dim), -1, dtype=np.int64
        )
        for node_index, label in enumerate(self.node_labels):
            for action, neighbor in enumerate(prior.valid_neighbors(label)):
                neighbor_index = prior.node_to_index[neighbor]
                key = (min(node_index, neighbor_index), max(node_index, neighbor_index))
                self.neighbor_node_indices[node_index, action] = neighbor_index
                self.neighbor_edge_indices[node_index, action] = self.edge_to_index[key]

        self._state: CentralState
        self._pose_graph: nx.MultiGraph
        self._done = False
        self._success = False
        self.initial_events: Tuple[FactorEvent, ...] = ()
        self.reset()

    @property
    def action_dim(self) -> int:
        return self.prior.action_dim

    @property
    def observation_dim(self) -> int:
        return self.num_edges + (3 + 2 * self.num_robots) * self.num_nodes + 1

    @property
    def observation_layout(self) -> dict:
        cursor = 0
        layout = {}
        for name, size in (
            ("self_traversals", self.num_edges), ("self_visited", self.num_nodes),
            ("self_position", self.num_nodes), ("team_visited", self.num_nodes),
            ("visitors", self.num_robots * self.num_nodes),
            ("representatives", self.num_robots * self.num_nodes), ("time", 1),
        ):
            layout[name] = (cursor, cursor + size)
            cursor += size
        return layout

    @property
    def state(self) -> CentralState:
        return self._state.clone()

    @property
    def pose_graph(self) -> nx.MultiGraph:
        return copy.deepcopy(self._pose_graph)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success(self) -> bool:
        return self._success

    @property
    def current_labels(self) -> Tuple[Any, ...]:
        return tuple(self.prior.index_to_node[int(index)] for index in self._state.positions)

    @property
    def derived_config(self) -> dict:
        return {
            "objective_version": OBJECTIVE_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "target_definition": "frozen_first_team_visit_keyframe_per_covered_region",
            "simultaneous_first_visit_rule": "lowest_robot_index",
            "motion_information": "one_summary_per_robot_undirected_edge",
            "inter_information": "one_summary_per_owner_visitor_region_star",
            "inter_noise_assumption": "independent_star_residuals_unvalidated_sensor_approximation",
            "known_initial_relative_poses": False,
            "omega_ref": self.omega_ref,
            "factor_catalog_counting": "robot_edge_and_pair_region_instances",
            "factor_catalog_size": self.num_robots * self.num_edges
            + len(self.closure_catalog),
            "traversal_raw_weights": self.edge_raw_weights.tolist(),
            "traversal_normalized_weights": self.edge_normalized_weights.tolist(),
            "pair_factor_weight_spec": self.pair_factor_weight_spec,
            "pair_raw_weight": self.pair_raw_weight,
            "pair_normalized_weight": self.pair_normalized_weight,
            "alpha_spec": self.alpha_spec,
            "alpha": self.alpha,
        }

    def schema(self) -> dict:
        return {
            "state_version": 2,
            "observation_dim": self.observation_dim,
            "reward_parameters": {"alpha": self.alpha, "beta": self.beta,
                                  "rho_v": self.rho_V, "rho_g": self.rho_g},
            "observation_layout": {key: list(value) for key, value in self.observation_layout.items()},
            "broadcast_identity_order": "self_then_cyclic_robot_indices",
            "node_labels": list(self.node_labels),
            "edge_endpoints": [list(edge) for edge in self.edge_endpoints],
            "edge_labels": [
                [self.prior.index_to_node[source], self.prior.index_to_node[target]]
                for source, target in self.edge_endpoints
            ],
            "robot_indices": list(self.robot_indices),
            "start_labels": list(self.start_labels),
            "start_positions": self.start_positions.tolist(),
            "closure_catalog": [list(item) for item in self.closure_catalog],
            "neighbor_nodes": self.neighbor_node_indices.tolist(),
            "neighbor_edges": self.neighbor_edge_indices.tolist(),
            "action_schema": self.prior.action_schema(),
            "t_max": self.t_max,
            "derived": self.derived_config,
        }

    def graph_tensor_spec(self, device: Any = None) -> dict:
        """Static shared-encoder inputs in canonical dense order."""

        target_device = torch.device("cpu" if device is None else device)
        return {
            "edge_endpoints": torch.as_tensor(
                self.edge_endpoints, dtype=torch.long, device=target_device
            ),
            "edge_omega": torch.as_tensor(
                self.edge_normalized_weights, dtype=torch.float32, device=target_device
            ),
            "edge_distance": torch.as_tensor(
                self.edge_distances, dtype=torch.float32, device=target_device
            ),
            "neighbor_node_indices": torch.as_tensor(
                self.neighbor_node_indices, dtype=torch.long, device=target_device
            ),
            "neighbor_edge_indices": torch.as_tensor(
                self.neighbor_edge_indices, dtype=torch.long, device=target_device
            ),
        }

    def _edge_support_visited(self, M_row: np.ndarray) -> np.ndarray:
        expected = np.zeros(self.num_nodes, dtype=np.bool_)
        for edge_index in np.flatnonzero(M_row > 0):
            source, target = self.edge_endpoints[int(edge_index)]
            expected[source] = True
            expected[target] = True
        return expected

    def validate_state(self, state: CentralState) -> None:
        if not isinstance(state, CentralState):
            raise TypeError("state must be a CentralState.")
        if state.M.shape != (self.num_robots, self.num_edges):
            raise ValueError(
                f"M must have shape {(self.num_robots, self.num_edges)}, "
                f"got {state.M.shape}."
            )
        if state.O.shape != (self.num_robots, self.num_nodes):
            raise ValueError(
                f"O must have shape {(self.num_robots, self.num_nodes)}, "
                f"got {state.O.shape}."
            )
        if state.positions.shape != (self.num_robots,):
            raise ValueError(
                f"positions must have shape {(self.num_robots,)}, "
                f"got {state.positions.shape}."
            )
        if state.owners.shape != (self.num_nodes,):
            raise ValueError("owners must contain one first-visitor entry per region.")
        covered = np.any(state.O, axis=0)
        if np.any(state.owners < -1) or np.any(state.owners >= self.num_robots):
            raise ValueError("Invalid first-visit owner index.")
        if not np.array_equal(state.owners >= 0, covered):
            raise ValueError("Exactly the covered regions must have frozen representatives.")
        for region in np.flatnonzero(covered):
            if not state.O[state.owners[region], region]:
                raise ValueError("A representative must be an actual visitor, never a broadcast-only visit.")
        for region in np.unique(self.start_positions):
            initial_owner = int(np.flatnonzero(self.start_positions == region)[0])
            if state.owners[region] != initial_owner:
                raise ValueError("An initial representative cannot be reassigned.")
        if state.time > self.t_max:
            raise ValueError("State time exceeds the absolute T_max horizon.")
        if np.any(state.M < 0):
            raise ValueError("Traversal counts M must be non-negative.")
        if not np.all(np.sum(state.M, axis=1, dtype=np.int64) == state.time):
            raise ValueError("Every robot must have exactly one traversal per elapsed joint step.")
        if np.any(state.positions < 0) or np.any(state.positions >= self.num_nodes):
            raise ValueError("A robot position is outside the canonical node index.")

        for robot in self.robot_indices:
            position = int(state.positions[robot])
            if not bool(state.O[robot, position]):
                raise ValueError(f"Robot {robot}'s current position must be marked visited.")
            if not bool(state.O[robot, self.start_positions[robot]]):
                raise ValueError(f"Robot {robot}'s start node must remain marked visited.")
            expected = self._edge_support_visited(state.M[robot])
            expected[self.start_positions[robot]] = True
            if not np.array_equal(state.O[robot], expected):
                raise ValueError(
                    f"Robot {robot}'s O row must equal its start plus traversal-edge support."
                )

            # Aggregate undirected counts must admit a trail from the fixed start
            # to the stored current position.  This catches impossible restores.
            degrees = np.zeros(self.num_nodes, dtype=np.int64)
            support = nx.Graph()
            support.add_node(int(self.start_positions[robot]))
            for edge_index, count in enumerate(state.M[robot]):
                if count <= 0:
                    continue
                source, target = self.edge_endpoints[edge_index]
                degrees[source] += int(count)
                degrees[target] += int(count)
                support.add_edge(source, target)
            odd = set(np.flatnonzero(degrees % 2).tolist())
            start = int(self.start_positions[robot])
            expected_odd = set() if start == position else {start, position}
            if odd != expected_odd:
                raise ValueError(
                    f"Robot {robot}'s traversal multiplicities cannot end at its position."
                )
            visited = set(np.flatnonzero(expected).tolist())
            if visited and not nx.is_connected(support.subgraph(visited)):
                raise ValueError(f"Robot {robot}'s traversal support is disconnected.")

    def _closure_set(self, state: CentralState) -> set:
        # Star centered on the frozen first visitor. Metadata alone is never O.
        return {
            (min(int(owner), robot), max(int(owner), robot), region)
            for region, owner in enumerate(state.owners)
            if owner >= 0
            for robot in self.robot_indices
            if robot != owner and state.O[robot, region]
        }

    def _add_motion_factor(
        self,
        graph: nx.MultiGraph,
        robot: int,
        edge_index: int,
        occurrence: int,
    ) -> None:
        source, target = self.edge_endpoints[edge_index]
        graph.add_node((robot, source), robot=robot, region_index=source)
        graph.add_node((robot, target), robot=robot, region_index=target)
        graph.add_edge(
            (robot, source),
            (robot, target),
            key=("motion", robot, edge_index, occurrence),
            factor_type="motion",
            robot=robot,
            edge_index=edge_index,
            occurrence=occurrence,
            raw_weight=float(self.edge_raw_weights[edge_index]),
            weight=float(self.edge_normalized_weights[edge_index]),
        )

    def _add_closure_factor(
        self,
        graph: nx.MultiGraph,
        robot_a: int,
        robot_b: int,
        region: int,
    ) -> None:
        closure_index = self.closure_to_index[(robot_a, robot_b, region)]
        graph.add_node((robot_a, region), robot=robot_a, region_index=region)
        graph.add_node((robot_b, region), robot=robot_b, region_index=region)
        graph.add_edge(
            (robot_a, region),
            (robot_b, region),
            key=("closure", closure_index),
            factor_type="closure",
            robot_pair=(robot_a, robot_b),
            region_index=region,
            closure_index=closure_index,
            raw_weight=float(self.pair_raw_weight),
            weight=float(self.pair_normalized_weight),
        )

    def build_pose_graph(self, state: Optional[CentralState] = None) -> nx.MultiGraph:
        """Rebuild unique motion summaries and owner-star closures from state."""

        selected = self._state if state is None else state
        self.validate_state(selected)
        graph = nx.MultiGraph()
        for robot in self.robot_indices:
            for region in np.flatnonzero(selected.O[robot]):
                graph.add_node(
                    (robot, int(region)), robot=robot, region_index=int(region)
                )
        # Fixed factor order: every motion factor first, then every closure.
        for robot in self.robot_indices:
            for edge_index in range(self.num_edges):
                if selected.M[robot, edge_index] > 0:
                    self._add_motion_factor(graph, robot, edge_index, 0)
        for robot_a, robot_b, region in sorted(self._closure_set(selected)):
            self._add_closure_factor(graph, robot_a, robot_b, region)
        return graph

    def component_count(self, state: Optional[CentralState] = None) -> int:
        graph = self.build_pose_graph(self._state if state is None else state)
        return nx.number_connected_components(graph)

    def structural_score(
        self,
        state: Optional[CentralState] = None,
        anchors: Optional[Mapping[frozenset, PoseNode]] = None,
    ) -> float:
        """Return ``alpha * sum_C logdet(cofactor(K_C))`` for representatives.

        Optional per-component anchors are exposed for anchor-invariance tests;
        they do not form part of environment state or reward.
        """

        selected = self._state if state is None else state
        graph = self.build_pose_graph(selected)
        return self._structural_score_graph(graph, selected.owners, anchors=anchors)

    def representative_nodes(self, state: Optional[CentralState] = None) -> Tuple[PoseNode, ...]:
        selected = self._state if state is None else state
        self.validate_state(selected)
        return tuple((int(owner), region) for region, owner in enumerate(selected.owners) if owner >= 0)

    def information_diagnostics(self, state: Optional[CentralState] = None) -> dict:
        selected = self._state if state is None else state
        graph = self.build_pose_graph(selected)
        targets = self.representative_nodes(selected)
        blocks = [component_representative_information(graph, component, targets)
                  for component in nx.connected_components(graph)]
        rep_score = self.alpha * sum(block["representative_logdet"] for block in blocks)
        conditional = self.alpha * sum(block["auxiliary_conditional_logdet"] for block in blocks)
        raw = self.alpha * sum(component_log_spanning_tree(graph, component)
                               for component in nx.connected_components(graph))
        return {
            "objective_version": OBJECTIVE_VERSION,
            "representative_score": float(rep_score),
            "raw_score_same_factor_graph": float(raw),
            "auxiliary_conditional_score": float(conditional),
            "decomposition_residual": float(raw - conditional - rep_score),
            "representative_count": len(targets),
            "auxiliary_count": graph.number_of_nodes() - len(targets),
            "components": len(blocks),
            "global_relative_covariance_available": len(blocks) == 1,
            "covariance_scope": "scalar_laplacian_surrogate_only",
            "owners": selected.owners.tolist(),
        }

    def _structural_score_graph(
        self, graph: nx.MultiGraph, owners: np.ndarray,
        anchors: Optional[Mapping[frozenset, PoseNode]] = None,
        allow_initial_auxiliary_singletons: bool = False,
    ) -> float:
        targets = {(int(owner), region) for region, owner in enumerate(owners) if owner >= 0}
        total = 0.0
        for component in nx.connected_components(graph):
            # Reset may temporarily contain an unconnected duplicate start before
            # its modeled co-location factor is processed. No dynamic state is
            # scored under this exception, and no reset reward is issued.
            if allow_initial_auxiliary_singletons and len(component) == 1 and not targets.intersection(component):
                continue
            anchor = None if anchors is None else anchors.get(frozenset(component))
            total += component_representative_information(
                graph, component, targets, anchor=anchor
            )["representative_logdet"]
        result = float(self.alpha * total)
        if not math.isfinite(result):
            raise FloatingPointError("The componentwise structural score is non-finite.")
        return result

    def coverage_count(self, state: Optional[CentralState] = None) -> int:
        selected = self._state if state is None else state
        self.validate_state(selected)
        return int(np.any(selected.O, axis=0).sum())

    def potential(self, state: Optional[CentralState] = None) -> float:
        selected = self._state if state is None else state
        structural = self.structural_score(selected)
        coverage = self.coverage_count(selected)
        components = self.component_count(selected)
        value = structural + self.rho_V * coverage - self.rho_g * (components - 1)
        if not math.isfinite(value):
            raise FloatingPointError("The gauge potential is non-finite.")
        return float(value)

    def is_success(self, state: Optional[CentralState] = None) -> bool:
        selected = self._state if state is None else state
        return (
            self.coverage_count(selected) == self.num_nodes
            and self.component_count(selected) == 1
        )

    def local_observations(self, state: Optional[CentralState] = None) -> np.ndarray:
        selected = self._state if state is None else state
        self.validate_state(selected)
        observations = np.zeros(
            (self.num_robots, self.observation_dim), dtype=np.float32
        )
        union = np.any(selected.O, axis=0)
        for robot in self.robot_indices:
            order = [(robot + offset) % self.num_robots for offset in self.robot_indices]
            position = np.zeros(self.num_nodes, dtype=np.float32)
            position[int(selected.positions[robot])] = 1.0
            owner_flags = np.asarray([selected.owners == index for index in order], dtype=np.float32)
            observations[robot] = np.concatenate((
                selected.M[robot], selected.O[robot], position, union,
                selected.O[order].ravel(), owner_flags.ravel(), [float(selected.time)],
            ))
        return observations

    def action_masks(self, state: Optional[CentralState] = None) -> np.ndarray:
        selected = self._state if state is None else state
        self.validate_state(selected)
        masks = []
        for position in selected.positions:
            label = self.prior.index_to_node[int(position)]
            masks.append(self.prior.action_mask(label).cpu().numpy())
        return np.stack(masks).astype(np.bool_, copy=False)

    def _event_formula_and_class(
        self,
        graph: nx.MultiGraph,
        source: PoseNode,
        target: PoseNode,
        normalized_weight: float,
    ) -> Tuple[str, float]:
        source_exists = source in graph
        target_exists = target in graph
        if not source_exists and not target_exists:
            raise ValueError("A new factor cannot introduce two unrelated pose nodes.")
        if not source_exists or not target_exists:
            return "E1", float(self.alpha * math.log(normalized_weight))
        if nx.has_path(graph, source, target):
            resistance = effective_resistance(graph, source, target)
            return "E2", float(
                self.alpha * math.log1p(normalized_weight * resistance)
            )
        return "E3", float(self.alpha * math.log(normalized_weight))

    def _apply_event_specs(
        self,
        graph: nx.MultiGraph,
        motion_specs: Sequence[Tuple[int, int, int]],
        closure_specs: Sequence[Tuple[int, int, int]],
        owners: np.ndarray,
        allow_initial_auxiliary_singletons: bool = False,
    ) -> Tuple[FactorEvent, ...]:
        events = []
        # Same-kind factors use canonical integer tuples as their lexicographic key.
        for robot, edge_index, occurrence in sorted(motion_specs):
            source_region, target_region = self.edge_endpoints[edge_index]
            source = (robot, source_region)
            target = (robot, target_region)
            normalized = float(self.edge_normalized_weights[edge_index])
            event_class, formula = self._event_formula_and_class(
                graph, source, target, normalized
            )
            before = self._structural_score_graph(graph, owners)
            self._add_motion_factor(graph, robot, edge_index, occurrence)
            after = self._structural_score_graph(graph, owners)
            events.append(
                FactorEvent(
                    order=len(events),
                    factor_type="motion",
                    event_class=event_class,
                    endpoints=(source, target),
                    normalized_weight=normalized,
                    raw_weight=float(self.edge_raw_weights[edge_index]),
                    structural_delta_formula=after - before,
                    raw_structural_delta_formula=formula,
                    robot=robot,
                    edge_index=edge_index,
                    occurrence=occurrence,
                )
            )
        for robot_a, robot_b, region in sorted(closure_specs):
            source = (robot_a, region)
            target = (robot_b, region)
            normalized = float(self.pair_normalized_weight)
            event_class, formula = self._event_formula_and_class(
                graph, source, target, normalized
            )
            before = self._structural_score_graph(graph, owners, allow_initial_auxiliary_singletons=allow_initial_auxiliary_singletons)
            self._add_closure_factor(graph, robot_a, robot_b, region)
            after = self._structural_score_graph(graph, owners, allow_initial_auxiliary_singletons=allow_initial_auxiliary_singletons)
            events.append(
                FactorEvent(
                    order=len(events),
                    factor_type="closure",
                    event_class=event_class,
                    endpoints=(source, target),
                    normalized_weight=normalized,
                    raw_weight=float(self.pair_raw_weight),
                    structural_delta_formula=after - before,
                    raw_structural_delta_formula=formula,
                    robot_pair=(robot_a, robot_b),
                    region_index=region,
                )
            )
        return tuple(events)

    def reset(self) -> Tuple[CentralState, np.ndarray, np.ndarray]:
        M = np.zeros((self.num_robots, self.num_edges), dtype=np.int64)
        O = np.zeros((self.num_robots, self.num_nodes), dtype=np.bool_)
        O[np.arange(self.num_robots), self.start_positions] = True
        owners = np.full(self.num_nodes, -1, dtype=np.int64)
        for robot, region in enumerate(self.start_positions):
            if owners[region] == -1:
                owners[region] = robot
        state = CentralState(M=M, O=O, positions=self.start_positions, time=0, owners=owners)
        self.validate_state(state)

        # Reset-time co-location closures obey the same pair-region uniqueness
        # and deterministic closure ordering as later visits.
        graph = nx.MultiGraph()
        for robot in self.robot_indices:
            graph.add_node(
                (robot, int(self.start_positions[robot])),
                robot=robot,
                region_index=int(self.start_positions[robot]),
            )
        self.initial_events = self._apply_event_specs(
            graph, motion_specs=(), closure_specs=sorted(self._closure_set(state)),
            owners=owners, allow_initial_auxiliary_singletons=True,
        )
        rebuilt = self.build_pose_graph(state)
        if not nx.utils.graphs_equal(graph, rebuilt):
            raise AssertionError("Reset factor processing does not match deterministic rebuild.")

        self._state = state
        self._pose_graph = rebuilt
        self._success = self.is_success(state)
        self._done = self._success or state.time >= self.t_max
        return state.clone(), self.local_observations(state), self.action_masks(state)

    def snapshot(self) -> CentralState:
        return self._state.clone()

    def restore(self, state: CentralState) -> Tuple[np.ndarray, np.ndarray]:
        restored = state.clone()
        self.validate_state(restored)
        self._state = restored
        self._pose_graph = self.build_pose_graph(restored)
        self._success = self.is_success(restored)
        self._done = self._success or restored.time >= self.t_max
        return self.local_observations(restored), self.action_masks(restored)

    def _decode_joint_actions(
        self, state: CentralState, actions: Sequence[Any]
    ) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
        if len(actions) != self.num_robots:
            raise ValueError(
                f"A joint action must contain exactly {self.num_robots} local actions."
            )
        action_indices = []
        next_positions = np.empty(self.num_robots, dtype=np.int64)
        edge_indices = np.empty(self.num_robots, dtype=np.int64)
        distances = np.empty(self.num_robots, dtype=np.float64)
        for robot, action in enumerate(actions):
            if isinstance(action, torch.Tensor):
                if action.numel() != 1:
                    raise ValueError("Each robot action tensor must be scalar.")
                action = action.item()
            if isinstance(action, (bool, np.bool_)) or not isinstance(
                action, (int, np.integer)
            ):
                raise TypeError("Every local action must be an integer index.")
            action = int(action)
            current_index = int(state.positions[robot])
            current_label = self.prior.index_to_node[current_index]
            next_label = self.prior.action_to_node(current_label, action)
            next_index = self.prior.node_to_index[next_label]
            if next_index == current_index:
                raise ValueError("Stay/wait actions are not part of the abstract model.")
            edge_index = int(self.neighbor_edge_indices[current_index, action])
            if edge_index < 0:
                raise ValueError("A padded/invalid action cannot enter a joint step.")
            action_indices.append(action)
            next_positions[robot] = next_index
            edge_indices[robot] = edge_index
            distances[robot] = self.edge_distances[edge_index]
        return tuple(action_indices), next_positions, edge_indices, distances

    def step(self, actions: Sequence[Any]) -> StepResult:
        """Execute exactly one atomic synchronous joint decision round."""

        if self._done:
            reason = "success" if self._success else "absolute T_max horizon"
            raise RuntimeError(f"The episode is terminal ({reason}); no implicit stay is allowed.")
        before_state = self._state
        action_indices, next_positions, edge_indices, distances = self._decode_joint_actions(
            before_state, actions
        )

        M = np.array(before_state.M, copy=True)
        O = np.array(before_state.O, copy=True)
        owners = np.array(before_state.owners, copy=True)
        motion_specs = []
        for robot in self.robot_indices:
            edge_index = int(edge_indices[robot])
            occurrence = int(M[robot, edge_index])
            M[robot, edge_index] += 1
            O[robot, int(next_positions[robot])] = True
            if owners[next_positions[robot]] == -1:
                owners[next_positions[robot]] = robot
            if occurrence == 0:
                motion_specs.append((robot, edge_index, 0))
        after_state = CentralState(
            M=M,
            O=O,
            positions=next_positions,
            time=before_state.time + 1,
            owners=owners,
        )
        self.validate_state(after_state)

        before_graph = self.build_pose_graph(before_state)
        event_graph = copy.deepcopy(before_graph)
        new_closures = sorted(
            self._closure_set(after_state) - self._closure_set(before_state)
        )
        events = self._apply_event_specs(event_graph, motion_specs, new_closures, owners)
        after_graph = self.build_pose_graph(after_state)
        if not nx.utils.graphs_equal(event_graph, after_graph):
            raise AssertionError("Incremental factor processing does not match state reconstruction.")

        structural_before = self._structural_score_graph(before_graph, before_state.owners)
        structural_after = self._structural_score_graph(after_graph, after_state.owners)
        if not math.isclose(sum(event.structural_delta_formula for event in events),
                            structural_after - structural_before, rel_tol=1e-8, abs_tol=1e-9):
            raise AssertionError("Representative event deltas must sum to the structural reward.")
        coverage_before = self.coverage_count(before_state)
        coverage_after = self.coverage_count(after_state)
        components_before = nx.number_connected_components(before_graph)
        components_after = nx.number_connected_components(after_graph)
        travel_distance = float(np.sum(distances))
        channels = RewardChannels(
            spectral=structural_after - structural_before,
            coverage=self.rho_V * (coverage_after - coverage_before),
            gauge=self.rho_g * (components_before - components_after),
            travel=-self.beta * travel_distance,
        )
        potential_before = (
            structural_before
            + self.rho_V * coverage_before
            - self.rho_g * (components_before - 1)
        )
        potential_after = (
            structural_after
            + self.rho_V * coverage_after
            - self.rho_g * (components_after - 1)
        )
        expected_without_travel = potential_after - potential_before
        actual_without_travel = channels.spectral + channels.coverage + channels.gauge
        if not math.isclose(
            expected_without_travel, actual_without_travel, rel_tol=1e-11, abs_tol=1e-11
        ):
            raise AssertionError("Gauge reward channels violate the potential difference.")

        success = coverage_after == self.num_nodes and components_after == 1
        done = success or after_state.time >= self.t_max
        self._state = after_state
        self._pose_graph = after_graph
        self._success = success
        self._done = done
        return StepResult(
            state=after_state.clone(),
            observations=self.local_observations(after_state),
            action_masks=self.action_masks(after_state),
            actions=action_indices,
            reward=channels.total,
            reward_channels=channels,
            done=done,
            success=success,
            events=events,
            structural_before=structural_before,
            structural_after=structural_after,
            potential_before=float(potential_before),
            potential_after=float(potential_after),
            components_before=components_before,
            components_after=components_after,
            travel_distance=travel_distance,
        )

    @staticmethod
    def telescoping_residual(
        initial_potential: float,
        final_potential: float,
        transitions: Sequence[StepResult],
        beta: float,
    ) -> float:
        """Numerical residual of the full-episode GCR telescoping identity."""

        observed = sum(result.reward for result in transitions)
        total_distance = sum(result.travel_distance for result in transitions)
        expected = float(final_potential - initial_potential - beta * total_distance)
        return float(observed - expected)


# Clear aliases for call sites that prefer the paper's terminology.
MultiRobotGaugeEnv = GEVDMultiRobotEnv
GaugeAwareMultiRobotEnv = GEVDMultiRobotEnv


__all__ = [
    "CentralState",
    "RewardChannels",
    "FactorEvent",
    "StepResult",
    "component_log_spanning_tree",
    "effective_resistance",
    "component_representative_information",
    "OBJECTIVE_VERSION",
    "OBSERVATION_VERSION",
    "GEVDMultiRobotEnv",
    "MultiRobotGaugeEnv",
    "GaugeAwareMultiRobotEnv",
]
