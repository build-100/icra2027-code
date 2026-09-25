"""Read-only evaluation and serialization helpers, independent of old workers."""
import copy
import hashlib
import random
from contextlib import contextmanager
import numpy as np
import torch
from gevd.training.base import _network_digest
from gevd.environments.prior_graph import PriorGraph
from gevd.environments.multi_robot import GEVDMultiRobotEnv


def object_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            visit(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item: visit(child)
        elif isinstance(item, (set, frozenset)):
            visit(sorted(item, key=repr))
        else:
            digest.update(repr(item).encode())
        digest.update(b"\0")
    visit(value)
    return digest.hexdigest()


def training_signature(tr):
    result = dict(online=_network_digest(tr.agent.online), target=_network_digest(tr.agent.target),
        optimizer=object_digest(tr.agent.optimizer.state_dict()),
        event=object_digest(tr.event_credit.state_dict()),
        sizes=[len(tr.factual_replay), len(tr.branch_replay)],
        counters=[tr.global_step, tr.episode_count, tr.branch_transition_count, tr.agent.update_steps],
        histories=[len(tr.loss_history), len(tr.event_loss_history), len(tr.mix_history),
                   len(tr.agent.diagnostic_history)],
        best=object_digest(tr.best_factual), retention=object_digest(tr.policy_retention.state_dict()))
    if hasattr(tr.agent, 'readonly_signature'):
        result['retrospective_utility'] = tr.agent.readonly_signature()
    return result


@contextmanager
def preserved_readonly(tr, ledger):
    state = tr.env.snapshot()
    signature = training_signature(tr)
    ledger_signature = [ledger.total, len(ledger.events), len(ledger.rollouts),
                        len(ledger.unique), object_digest(ledger.best)]
    rng = [random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           tr.agent.rng.getstate(), tr.branch_rng.getstate(), tr.branch_behavior_rng.getstate()]
    modules = [(module, module.training) for network in (tr.agent.online, tr.agent.target)
               for module in network.modules()]
    try:
        yield
    finally:
        tr.env.restore(state)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng[3])
        tr.agent.rng.setstate(rng[4]); tr.branch_rng.setstate(rng[5]); tr.branch_behavior_rng.setstate(rng[6])
        for module, training in modules: module.training = training
        assert training_signature(tr) == signature, "Read-only evaluation mutated training state"
        assert [ledger.total, len(ledger.events), len(ledger.rollouts), len(ledger.unique),
                object_digest(ledger.best)] == ledger_signature, "Evaluation entered search candidates"


def simple_episode(result, episode, factual_steps, beta, nodes):
    success = bool(result["success"])
    terminal = bool(result["terminal"])
    coverage, components = int(result["final_coverage_count"]), int(result["final_component_count"])
    if not terminal:
        failure = "budget_truncated"
    elif success:
        failure = None
    elif coverage == nodes:
        failure = "full_unfused"
    elif components == 1:
        failure = "missed_coverage"
    else:
        failure = "both"
    raw_j = float(result["final_structural_score"] - beta * result["total_distance"])
    return dict(episode=episode, factual_steps=factual_steps, routes=result["routes"],
        success=success, terminal=terminal, budget_truncated=bool(result["budget_truncated"]),
        failure_type=failure, strict_terminal_sample=terminal,
        S=float(result["final_structural_score"]), D=float(result["total_distance"]),
        T=int(result["joint_steps"]), J=raw_j, J_raw=raw_j, J_success=raw_j if success else None,
        coverage=coverage, V=int(nodes), c=components, terminal_penalty=float(result["reward_channels"].get("terminal_penalty", 0.0)), G=float(result["return"]), objective_initial=float(result["initial_potential"]), objective_terminal=float(result["final_potential"] - beta * result["total_distance"]), **{"return": float(result["return"])})


def replay_metrics(config, routes, require_success=False):
    settings, reward = config["environment"], config["reward"]
    prior = PriorGraph(settings["graph_path"], settings["map_width"], settings["start_nodes"][0], "cpu")
    env = GEVDMultiRobotEnv(prior, tuple(settings["start_nodes"]), settings["t_max"],
        reward["alpha"], reward["beta"], reward["rho_v"], reward["rho_g"], reward["pair_factor_weight"])
    distance = np.zeros(env.num_robots); events = []; timeline = []
    for t in range(len(routes[0]) - 1):
        assert tuple(route[t] for route in routes) == env.current_labels
        actions = [env.prior.action_for_neighbor(route[t], route[t + 1]) for route in routes]
        for i, action in enumerate(actions):
            distance[i] += env.edge_distances[env.neighbor_edge_indices[env.state.positions[i], action]]
        result = env.step(actions)
        for event in result.events:
            if event.event_class in ("E2", "E3"):
                events.append(dict(t=t + 1, type=event.factor_type, **{"class": event.event_class},
                    robot=event.robot, pair=event.robot_pair,
                    region=env.node_labels[event.region_index] if event.region_index is not None else None,
                    delta_S=event.structural_delta_formula))
        timeline.append(dict(t=t + 1, coverage=env.coverage_count(), c=env.component_count(), S=env.structural_score()))
    if require_success: assert env.success
    visits = env.state.O.sum(axis=0)
    return dict(success=bool(env.success), coverage=env.coverage_count(), V=env.num_nodes,
        c=env.component_count(), T=env.state.time, D=float(distance.sum()), S=env.structural_score(),
        J=env.structural_score() - env.beta * float(distance.sum()),
        overlap=int((visits >= 2).sum()), robot_distance=distance.tolist(),
        robot_unique_regions=env.state.O.sum(axis=1).tolist(), routes=routes, events=events, timeline=timeline)
