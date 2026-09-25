"""Focused regressions for the one active DQN/VDN/MCBR learning path."""

from __future__ import annotations

import copy
import itertools
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from gevd.models.dqn import DQNAgent
from gevd.training.base import VDNTrainer, _network_digest, resolve_inside_project
from gevd.environments.prior_graph import PriorGraph
from gevd.environments.multi_robot import GEVDMultiRobotEnv
from gevd.models.networks import GraphUtilityNetwork
from gevd.training.replay import MixedReplaySampler, TransitionReplayBuffer


def make_graph(labels=(10, 30, 70, 90), edges=None):
    """Create a weighted in-memory graph; no map3/map4/map7 dependency."""

    if edges is None:
        edges = ((10, 30), (30, 70), (70, 90))
    graph = nx.Graph()
    for index, label in enumerate(labels):
        graph.add_node(label, position=(float(index), float(index % 2)))
    for index, (source, target) in enumerate(edges):
        graph.add_edge(
            source,
            target,
            weight=float(index + 1),
            distance=float(index + 1),
            omega=float(index + 2),
        )
    return graph


def make_prior(labels=(10, 30, 70, 90), edges=None, start=10):
    return PriorGraph(graph=make_graph(labels, edges), start_node=start, device="cpu")


def make_env(robot_count=2, t_max=4):
    starts = (10, 90) if robot_count == 2 else (10,)
    return GEVDMultiRobotEnv(
        make_prior(start=starts[0]),
        starts,
        t_max=t_max,
        alpha="reciprocal_num_regions",
        beta=0.1,
        rho_V=2.0,
        rho_g=3.0,
        pair_factor_weight="median_traversal",
    )


def make_config(robot_count=2, t_max=4, seed=17):
    starts = [10, 90] if robot_count == 2 else [10]
    return {
        "environment": {
            # The direct-constructor tests never read this file, but checkpoint
            # validation still verifies that the declared path stays inside the project.
            "graph_path": "tests/unused-in-memory-graph",
            "map_width": 1.0,
            "robot_count": robot_count,
            "start_nodes": starts,
            "t_max": t_max,
        },
        "reward": {
            "alpha": "reciprocal_num_regions",
            "beta": 0.1,
            "rho_v": 2.0,
            "rho_g": 3.0,
            "pair_factor_weight": "median_traversal",
            "catalog_counting": "robot_edge_and_pair_region_instances",
        },
        "vdn": {
            "episodes": 1,
            "hidden_dim": 16,
            "gamma": 1.0,
            "learning_rate": 0.001,
            "batch_size": 2,
            "factual_replay_capacity": 64,
            "min_factual_replay": 2,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 10,
            "target_update_interval": 2,
            "updates_per_episode": 1,
            "gradient_clip": 10.0,
            "seed": seed,
            "device": "cpu",
        },
        "mcbr": {
            "branch_replay_capacity": 64,
            "branch_roots_per_episode": 2,
            "branch_fraction": 0.5,
        },
        "guarantee": {
            "mode": "empirical",
            "delta_s_bound": None,
            "successful_reference_distance": None,
            "strict_margin": 1.0,
        },
        "output": {
            "directory": "results/generated/focused_vdn_tests",
            "checkpoint_name": "checkpoint.pt",
            "summary_name": "summary.json",
            "routes_name": "routes.json",
        },
    }


def zero_online(agent):
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
    agent.sync_target()


def push_transition(buffer, robots=1, obs_dim=2, action_dim=3, reward=1.0, done=True):
    state = np.zeros((robots, obs_dim), dtype=np.float32)
    next_state = np.ones((robots, obs_dim), dtype=np.float32)
    actions = np.zeros(robots, dtype=np.int64)
    masks = np.zeros((robots, action_dim), dtype=np.bool_)
    next_masks = np.ones((robots, action_dim), dtype=np.bool_) if done else masks
    buffer.push(state, actions, reward, next_state, masks, next_masks, done)


class FixedLinearQ(nn.Module):
    """Tiny controllable Q network used to expose the VDN arithmetic."""

    def __init__(self, weights):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        self.linear = nn.Linear(weights.shape[1], weights.shape[0], bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weights)

    def forward(self, observation):
        return self.linear(observation)


class DQNPrimitiveTests(unittest.TestCase):
    def make_agent(self, action_dim=3, gamma=0.5, target_interval=2, loss_kind="huber"):
        return DQNAgent(
            obs_dim=4,
            action_dim=action_dim,
            hidden_dim=8,
            learning_rate=0.01,
            gamma=gamma,
            target_update_interval=target_interval,
            gradient_clip=10.0,
            seed=3,
            device="cpu",
            loss_kind=loss_kind,
        )

    @staticmethod
    def set_constant_q(network, values):
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.q_values[-1].bias.copy_(torch.tensor(values, dtype=torch.float32))

    def test_mask_tensor_accepts_an_existing_tensor(self):
        source = torch.tensor([[False, True]], dtype=torch.bool)
        converted = DQNAgent._mask_tensor(source, torch.device("cpu"))
        self.assertEqual(converted.device.type, "cpu")
        self.assertEqual(converted.dtype, torch.bool)
        self.assertTrue(torch.equal(converted, source))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_learning_keeps_gradients_and_masks_on_cuda(self):
        agent = DQNAgent(
            obs_dim=2,
            action_dim=2,
            hidden_dim=4,
            learning_rate=0.01,
            gamma=1.0,
            target_update_interval=2,
            gradient_clip=10.0,
            seed=3,
            device="cuda",
        )
        replay = TransitionReplayBuffer(4, seed=5)
        push_transition(replay, robots=1, obs_dim=2, action_dim=2)
        loss = agent.learn(replay.sample(1))
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.last_gradient_device, "cuda:0")

    def test_deterministic_neighbor_mapping_and_invalid_action_selection(self):
        prior = make_prior()
        self.assertEqual(prior.node_labels, (10, 30, 70, 90))
        for node in prior.node_labels:
            neighbors = prior.valid_neighbors(node)
            mask = prior.action_mask(node)
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(int((~mask).sum()), len(neighbors))
            for action, neighbor in enumerate(neighbors):
                self.assertEqual(prior.action_to_node(node, action), neighbor)
                self.assertEqual(prior.action_for_neighbor(node, neighbor), action)

        agent = self.make_agent()
        self.set_constant_q(agent.online, [1.0, 100.0, -2.0])
        invalid = np.array([False, True, False], dtype=np.bool_)
        self.assertEqual(agent.select_action(np.zeros(4), invalid, epsilon=0.0), 0)
        sampled = {
            agent.select_action(np.zeros(4), invalid, epsilon=1.0)
            for _ in range(100)
        }
        self.assertEqual(sampled, {0, 2})

    def test_terminal_does_not_bootstrap_and_legal_mask_controls_team_max(self):
        agent = self.make_agent(gamma=0.5)
        self.set_constant_q(agent.target, [2.0, 100.0, 3.0])
        next_states = np.zeros((2, 2, 4), dtype=np.float32)
        next_masks = np.array(
            [
                [[True, True, True], [True, True, True]],
                [[False, True, False], [False, True, False]],
            ],
            dtype=np.bool_,
        )
        targets = agent.compute_td_targets(
            rewards=[5.0, 1.0],
            next_states=next_states,
            next_invalid_masks=next_masks,
            dones=[True, False],
        )
        # Terminal all-invalid masks are never evaluated.  On the active row the
        # invalid Q=100 is excluded and each robot contributes max(2,3)=3.
        np.testing.assert_allclose(targets.cpu().numpy(), [5.0, 4.0])
        with self.assertRaises(ValueError):
            agent.compute_td_targets(
                [0.0], np.zeros((1, 2, 4), np.float32),
                np.ones((1, 2, 3), np.bool_), [False]
            )

    def test_positive_td_reward_scaling_preserves_bootstrap_structure(self):
        agent = DQNAgent(
            obs_dim=4, action_dim=2, hidden_dim=8, learning_rate=0.01,
            gamma=1.0, target_update_interval=2, gradient_clip=10.0,
            seed=3, device="cpu", td_reward_scale=0.1,
        )
        self.set_constant_q(agent.target, [2.0, 3.0])
        targets = agent.compute_td_targets(
            rewards=[10.0, 10.0],
            next_states=np.zeros((2, 1, 4), dtype=np.float32),
            next_invalid_masks=np.array(
                [[[False, False]], [[True, True]]], dtype=np.bool_
            ),
            dones=[False, True],
        )
        np.testing.assert_allclose(targets.cpu().numpy(), [4.0, 1.0])

    def test_target_sync_uses_optimizer_update_cadence(self):
        agent = self.make_agent(action_dim=2, gamma=1.0, target_interval=2)
        replay = TransitionReplayBuffer(4, seed=0)
        for reward in (1.0, 2.0):
            replay.push(
                np.zeros(4, np.float32), 0, reward, np.ones(4, np.float32),
                [False, True], [True, True], True,
            )
        initial_target = copy.deepcopy(agent.target.state_dict())
        agent.learn(replay.sample(2))
        for key, value in agent.target.state_dict().items():
            self.assertTrue(torch.equal(value, initial_target[key]))
        agent.learn(replay.sample(2))
        for online, target in zip(agent.online.parameters(), agent.target.parameters()):
            self.assertTrue(torch.equal(online, target))
        self.assertEqual(agent.target_sync_steps, [2])
        self.assertEqual(agent.last_target_sync_update, 2)
        self.assertEqual(len(agent.diagnostic_history), 2)
        self.assertTrue(agent.diagnostic_history[-1]["target_synced"])
        self.assertTrue(np.isfinite(agent.diagnostic_history[-1]["td_abs_mean"]))

    def test_joint_replay_shapes_and_dtypes(self):
        replay = TransitionReplayBuffer(4, seed=1)
        for reward in (1.5, 2.5):
            push_transition(replay, robots=2, obs_dim=4, action_dim=3, reward=reward)
        batch = replay.sample(2)
        self.assertEqual(np.stack(batch.state).shape, (2, 2, 4))
        self.assertEqual(np.stack(batch.state).dtype, np.float32)
        self.assertEqual(np.asarray(batch.action).shape, (2, 2))
        self.assertEqual(np.asarray(batch.action).dtype, np.int64)
        self.assertEqual(np.asarray(batch.reward).dtype.kind, "f")
        self.assertEqual(np.stack(batch.action_mask).dtype, np.bool_)
        self.assertEqual(np.asarray(batch.done).dtype, np.bool_)


class UnifiedVDNArithmeticTests(unittest.TestCase):
    @staticmethod
    def make_linear_agent(loss_kind="mse"):
        # Robot observation e0 -> [1,4,2]; e1 -> [3,0,5].
        network = FixedLinearQ([[1.0, 3.0], [4.0, 0.0], [2.0, 5.0]])
        return DQNAgent(
            obs_dim=2,
            action_dim=3,
            hidden_dim=4,
            learning_rate=0.001,
            gamma=1.0,
            target_update_interval=20,
            gradient_clip=None,
            seed=5,
            device="cpu",
            network=network,
            loss_kind=loss_kind,
        )

    def test_n1_is_the_same_shared_learner_and_has_no_fake_robot_closure(self):
        agent = self.make_linear_agent()
        state = np.array([[1.0, 0.0]], dtype=np.float32)
        mask = np.array([[False, False, False]], dtype=np.bool_)
        action = agent.select_actions(state, mask, epsilon=0.0)
        self.assertEqual(action.tolist(), [1])
        local = agent.online(torch.as_tensor(state)).detach().numpy()[0, action[0]]
        self.assertEqual(float(local), 4.0)

        env = make_env(robot_count=1, t_max=2)
        _, observations, masks = env.reset()
        self.assertEqual(observations.shape, (1, env.observation_dim))
        self.assertEqual(masks.shape, (1, env.action_dim))
        self.assertEqual(env.closure_catalog, ())
        result = env.step([0])
        self.assertEqual(result.reward_channels.gauge, 0.0)
        self.assertFalse(any(event.factor_type == "closure" for event in result.events))

    def test_vdn_sum_and_independent_argmax_equal_cartesian_argmax(self):
        agent = self.make_linear_agent()
        observations = np.eye(2, dtype=np.float32)
        masks = np.array(
            [[False, True, False], [False, False, True]], dtype=np.bool_
        )
        q = agent.online(torch.as_tensor(observations)).detach().numpy()
        actions = agent.select_actions(observations, masks, epsilon=0.0)
        self.assertEqual(actions.tolist(), [2, 0])

        valid = [np.flatnonzero(~row).tolist() for row in masks]
        scored = {
            joint: sum(q[robot, action] for robot, action in enumerate(joint))
            for joint in itertools.product(*valid)
        }
        brute_force = max(scored, key=scored.get)
        self.assertEqual(tuple(actions), brute_force)
        self.assertEqual(
            scored[tuple(actions)],
            sum(float(q[i, actions[i]]) for i in range(2)),
        )

    def test_one_team_squared_td_error_not_one_loss_per_robot(self):
        agent = self.make_linear_agent(loss_kind="mse")
        online_calls = []
        hook = agent.online.register_forward_hook(
            lambda _module, inputs, _output: online_calls.append(tuple(inputs[0].shape))
        )
        replay = TransitionReplayBuffer(2, seed=0)
        states = np.eye(2, dtype=np.float32)
        actions = np.array([1, 2], dtype=np.int64)  # q_team = 4 + 5 = 9
        masks = np.zeros((2, 3), dtype=np.bool_)
        replay.push(
            states, actions, 12.0, states, masks,
            np.ones((2, 3), dtype=np.bool_), True,
        )
        try:
            loss = agent.learn(replay.sample(1))
        finally:
            hook.remove()
        self.assertAlmostEqual(loss, (12.0 - 9.0) ** 2)
        self.assertEqual(agent.last_td_errors.shape, (1,))
        self.assertAlmostEqual(float(agent.last_td_errors[0]), 3.0)
        # Both robots were flattened through one shared online module invocation.
        self.assertEqual(online_calls, [(2, 2)])

    def test_configurable_residual_graph_encoder_preserves_local_input_boundary(self):
        config = make_config(robot_count=2, t_max=4, seed=9)
        config["vdn"].update(
            {
                "graph_message_passing_layers": 3,
                "graph_residual_updates": True,
                "normalize_local_features": True,
                "network_initialization": "relu_orthogonal",
            }
        )
        trainer = VDNTrainer(make_env(robot_count=2, t_max=4), config)
        network = trainer.agent.online
        self.assertIsInstance(network, GraphUtilityNetwork)
        self.assertEqual(network.message_passing_layers, 3)
        self.assertTrue(network.residual_updates)
        self.assertTrue(network.normalize_local_features)
        self.assertEqual(network.t_max, 4)
        self.assertEqual(len(network.extra_messages), 2)
        self.assertEqual(len(network.residual_norms), 3)

        state, observations, masks = trainer.env.reset()
        # Environment semantics remain raw (M_i, O_i, v_i, t); normalization is
        # strictly internal to the shared local graph utility.
        self.assertEqual(float(observations[0, -1]), float(state.time))
        q_values = network(torch.as_tensor(observations, dtype=torch.float32))
        self.assertEqual(q_values.shape, (2, trainer.env.action_dim))
        self.assertTrue(torch.isfinite(q_values).all())
        self.assertEqual(masks.shape, (2, trainer.env.action_dim))
        self.assertFalse(any("central" in key for key in network.state_dict()))


class MixedReplayTests(unittest.TestCase):
    def test_fixed_eta_and_empty_branch_fallback(self):
        factual = TransitionReplayBuffer(16, seed=1)
        branch = TransitionReplayBuffer(16, seed=2)
        for value in range(8):
            push_transition(factual, reward=float(value))
            push_transition(branch, reward=float(100 + value))
        sampler = MixedReplaySampler(0.5, seed=3)
        batch, counts = sampler.sample(factual, branch, 6)
        self.assertEqual(counts, {"factual": 3, "branch": 3})
        self.assertEqual(sum(reward >= 100.0 for reward in batch.reward), 3)

        empty = TransitionReplayBuffer(4, seed=4)
        batch, counts = sampler.sample(factual, empty, 6)
        self.assertEqual(counts, {"factual": 6, "branch": 0})
        self.assertTrue(all(reward < 100.0 for reward in batch.reward))


class TrainerMCBRTests(unittest.TestCase):
    def make_trainer(self, seed=17):
        trainer = VDNTrainer(make_env(robot_count=2, t_max=4), make_config(seed=seed))
        # Equal utilities make the factual policy choose the lowest legal action,
        # yielding two deterministic alternative roots at absolute time one.
        zero_online(trainer.agent)
        return trainer

    @staticmethod
    def trace_signature(traces):
        return [
            {
                "root": trace["root"],
                "root_action": trace["branch_root_action"],
                "start": trace["absolute_start_time"],
                "end": trace["absolute_end_time"],
                "digest": trace["frozen_network_digest"],
                "records": [
                    (
                        record["actions"],
                        record["after_state"].M.tolist(),
                        record["after_state"].O.tolist(),
                        record["after_state"].positions.tolist(),
                        record["after_state"].time,
                        record["reward"],
                        record["done"],
                    )
                    for record in trace["records"]
                ],
            }
            for trace in traces
        ]

    def test_factual_only_roots_root_substitution_and_model_regeneration(self):
        trainer = self.make_trainer()
        factual = trainer.run_factual_episode(training=True)
        cache = factual["pre_action_cache"]
        roots_before = trainer.enumerate_branch_roots(cache)
        self.assertEqual(roots_before, [(1, 0, 1), (1, 1, 1)])
        factual_replay_size = len(trainer.factual_replay)
        online_digest = _network_digest(trainer.agent.online)

        traces = trainer.generate_branches(cache, maximum=2)
        self.assertEqual(len(traces), 2)
        self.assertGreater(len(trainer.branch_replay), 0)
        self.assertEqual(len(trainer.factual_replay), factual_replay_size)
        self.assertEqual(trainer.enumerate_branch_roots(), roots_before)
        self.assertEqual(_network_digest(trainer.agent.online), online_digest)

        for trace in traces:
            root_index, changed_robot, alternative = trace["root"]
            factual_root = cache[root_index]
            root_action = trace["branch_root_action"]
            changed = [
                robot
                for robot in range(trainer.env.num_robots)
                if root_action[robot] != factual_root.factual_action[robot]
            ]
            self.assertEqual(changed, [changed_robot])
            self.assertEqual(root_action[changed_robot], alternative)
            for robot in range(trainer.env.num_robots):
                if robot != changed_robot:
                    self.assertEqual(root_action[robot], factual_root.factual_action[robot])

            first_record = trace["records"][0]
            # Independently restore the factual root and execute the substituted
            # action: state, reward, channels and masks must be regenerated.
            verifier = make_env(robot_count=2, t_max=4)
            observations, masks = verifier.restore(factual_root.state)
            expected = verifier.step(root_action)
            self.assertEqual(first_record["before_state"], factual_root.state)
            self.assertEqual(first_record["after_state"], expected.state)
            self.assertAlmostEqual(first_record["reward"], expected.reward)
            self.assertEqual(first_record["reward_channels"], expected.reward_channels.as_dict())
            np.testing.assert_array_equal(first_record["action_masks"], masks)

            # The substituted successor is different from the factual future;
            # it was not copied from factual transition t.
            factual_successor = factual["transitions"][root_index].state
            self.assertNotEqual(first_record["after_state"], factual_successor)
            self.assertEqual(trace["absolute_start_time"], factual_root.absolute_time)
            self.assertLessEqual(trace["absolute_end_time"], trainer.env.t_max)
            self.assertEqual(
                len(trace["records"]),
                trace["absolute_end_time"] - trace["absolute_start_time"],
            )
            self.assertEqual(trace["frozen_network_digest"], online_digest)

        # Generated transitions are held only in D_B and never become roots.
        self.assertEqual(len(trainer.last_factual_cache), len(cache))
        self.assertTrue(all(root[0] < len(cache) for root in trainer.enumerate_branch_roots()))

    def test_fixed_seed_reproduces_roots_and_branch_rollouts(self):
        first = self.make_trainer(seed=29)
        first_episode = first.run_factual_episode(training=True)
        first_traces = first.generate_branches(first_episode["pre_action_cache"], maximum=2)

        second = self.make_trainer(seed=29)
        second_episode = second.run_factual_episode(training=True)
        second_traces = second.generate_branches(second_episode["pre_action_cache"], maximum=2)
        self.assertEqual(
            self.trace_signature(first_traces),
            self.trace_signature(second_traces),
        )

    def test_checkpoint_roundtrip_restores_both_replays_and_path_guard(self):
        trainer = self.make_trainer(seed=41)
        episode = trainer.run_factual_episode(training=True)
        trainer.generate_branches(episode["pre_action_cache"], maximum=2)
        # Include optimizer/update state in the round trip.
        trainer._updates_if_ready()
        checkpoint = PROJECT_ROOT / "results/generated" / "focused_vdn_tests" / "roundtrip.pt"
        try:
            trainer.save_checkpoint(checkpoint)
            restored = self.make_trainer(seed=41)
            restored.load_checkpoint(checkpoint)
            self.assertEqual(len(restored.factual_replay), len(trainer.factual_replay))
            self.assertEqual(len(restored.branch_replay), len(trainer.branch_replay))
            self.assertEqual(restored.episode_count, trainer.episode_count)
            self.assertEqual(restored.global_step, trainer.global_step)
            self.assertEqual(restored.branch_transition_count, trainer.branch_transition_count)
            self.assertEqual(restored.agent.update_steps, trainer.agent.update_steps)
            self.assertEqual(restored.config["effective"], trainer.config["effective"])
            self.assertEqual(restored.mixed_sampler.last_counts, trainer.mixed_sampler.last_counts)
            for original, loaded in zip(
                trainer.agent.online.parameters(), restored.agent.online.parameters()
            ):
                self.assertTrue(torch.equal(original, loaded))
            for original, loaded in zip(
                trainer.agent.target.parameters(), restored.agent.target.parameters()
            ):
                self.assertTrue(torch.equal(original, loaded))
        finally:
            if checkpoint.exists():
                checkpoint.unlink()
            parent = checkpoint.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

        with self.assertRaises(ValueError):
            resolve_inside_project(PROJECT_ROOT.parent / "outside-checkpoint.pt")

    def test_exact_factual_budget_uses_nonterminal_tail_without_branching_it(self):
        config = make_config(seed=43)
        config["vdn"]["updates_per_episode"] = 0
        config["vdn"]["updates_per_factual_step"] = 0.0
        config["mcbr"]["branch_roots_per_episode"] = 0
        trainer = VDNTrainer(make_env(robot_count=2, t_max=4), config)
        zero_online(trainer.agent)
        summaries = trainer.train_to_factual_budget(5)
        self.assertEqual(trainer.global_step, 5)
        self.assertEqual(len(trainer.factual_replay), 5)
        self.assertTrue(summaries[-1]["budget_truncated"])
        self.assertFalse(summaries[-1]["terminal"])
        self.assertEqual(summaries[-1]["branch_roots"], [])
        self.assertEqual(summaries[-1]["branch_transitions"], 0)

    def test_factual_budget_progress_callback_observes_completed_rollouts(self):
        config = make_config(seed=47)
        config["vdn"]["updates_per_episode"] = 0
        config["vdn"]["updates_per_factual_step"] = 0.0
        config["mcbr"]["branch_roots_per_episode"] = 0
        config["environment"]["t_max"] = 3
        trainer = VDNTrainer(make_env(robot_count=2, t_max=3), config)
        observed = []

        def callback(active_trainer, summary):
            observed.append((active_trainer.global_step, summary["joint_steps"]))

        summaries = trainer.train_to_factual_budget(4, progress_callback=callback)
        self.assertEqual(len(observed), len(summaries))
        self.assertEqual(observed[-1][0], 4)
        self.assertTrue(all(step > 0 for _, step in observed))


if __name__ == "__main__":
    unittest.main()
