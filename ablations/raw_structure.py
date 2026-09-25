"""B without representative-pose marginalization: raw structure in TD replay.

Only the learner's spectral increment changes to Delta S_raw (paper Eq. 5).
The simulator, candidate verifier/selector and evaluation retain common S/G.
Thus new raw information cannot masquerade as an improved common score.
"""
import math
import networkx as nx
import numpy as np
from gevd.training.search import BudgetedSearchTrainer
from gevd.environments.multi_robot import component_log_spanning_tree


def raw_structural_score(env, state=None):
    state = env.state if state is None else state
    graph = env.build_pose_graph(state)
    representatives = set(env.representative_nodes(state))
    return float(env.alpha * sum(component_log_spanning_tree(graph, component,
        anchor=sorted(representatives.intersection(component))[0])
        for component in nx.connected_components(graph)))


class RawStructureTrainer(BudgetedSearchTrainer):
    def __init__(self, env, config):
        if config.get('training_structure') != 'componentwise_raw':
            raise ValueError('Raw structure ablation requires training_structure=componentwise_raw.')
        if (config.get('retrospective_utility', {}).get('enabled', False)
                or config.get('event_aligned', {}).get('enabled', False)
                or config.get('policy_retention', {}).get('enabled', False)
                or config['mcbr']['branch_roots_per_episode'] or config['mcbr']['branch_fraction']):
            raise ValueError('This is an ablation on B: no auxiliary utility or alternative search.')
        super().__init__(env, config)
        self.config['effective']['implementation_choices']['training_structure'] = dict(
            score='componentwise raw full pose-graph log spanning tree (Eq. 5)',
            replay_reward='delta S_raw + rho_v delta n + rho_g component_reduction - beta distance',
            evaluation_score='representative Schur score S (Eq. 4)',
            candidate_selection='common S-beta*D among verified successful factual paths',
            original_graph_and_ownership_preserved=True)

    @property
    def algorithm_name(self):
        return 'B w/o Representative-Pose Marginalization'

    def _push(self, replay, observations, actions, result, masks):
        current_raw = raw_structural_score(self.env, result.state)
        raw_delta = current_raw - self._previous_raw_score
        channels = result.reward_channels.as_dict()
        channels['spectral'] = raw_delta
        training_reward = sum(channels.values())
        replay.push(observations, np.asarray(actions, dtype=np.int64), training_reward,
            result.observations, masks,
            self._terminal_masks(result.action_masks) if result.done else result.action_masks,
            result.done, reward_channels=channels)
        self._raw_training_return += training_reward
        self._previous_raw_score = current_raw

    def run_factual_episode(self, training=True, max_steps=None):
        if training:
            # All specified entry regions are distinct singleton components.
            self._previous_raw_score = 0.0
            self._raw_training_return = 0.0
        episode = super().run_factual_episode(training=training, max_steps=max_steps)
        raw = raw_structural_score(self.env)
        native_return = (episode['return'] + raw - episode['final_structural_score'])
        if training and not math.isclose(native_return, self._raw_training_return, abs_tol=1e-8):
            raise AssertionError('Raw training rewards do not telescope to the raw objective.')
        episode['raw_structure'] = dict(S_raw=raw, training_return=native_return,
            common_S=episode['final_structural_score'], common_G=episode['return'],
            training_replay_modified=training)
        return episode
