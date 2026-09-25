"""Single-component ablations of GEVD; environment/evaluation stay common.

B.1: parameter-shared independent Q learning with the full team reward for
each robot, mean local TD loss, one optimizer step per original joint batch.
A.2: remove only the explicit gauge increment in primary factual TD replay.
A.1: replace only primary replay's spectral increment by componentwise S_raw.
All three retain the independent bounded retrospective auxiliary branch.
"""
import math
from types import SimpleNamespace
import numpy as np
from gevd.training.gevd import GEVDTrainer
from ablations.raw_structure import raw_structural_score
from gevd.training.base import load_torch_checkpoint, resolve_inside_project

LABELS = {
    'no_vdn': 'GEVD w/o Collective Value Decomposition',
    'no_gauge': 'GEVD w/o Explicit Gauge-Nullity Term',
    'raw_structure': 'GEVD w/o Representative-Pose Marginalization',
}


class IndependentTDPrimary:
    """Reuse the unchanged masked DDQN update over singleton-robot samples.

Flattening B joint transitions into B*N local samples makes each target
0.1*r_team + gamma*Q_target(o_i',argmax Q_online(o_i')). No Q from another
robot enters that target or residual. The loss averages B*N residuals;
reward is not divided by N. Network, optimizer and initialization are reused.
"""
    def __init__(self, primary):
        self.base = primary

    def __getattr__(self, name):
        return getattr(self.base, name)

    @staticmethod
    def local_batch(batch):
        states = np.stack(batch.state)
        if states.ndim != 3:
            raise ValueError('Independent TD requires a joint [batch, robots, obs] batch.')
        batch_size, robots, width = states.shape
        return SimpleNamespace(
            state=states.reshape(batch_size * robots, width),
            next_state=np.stack(batch.next_state).reshape(batch_size * robots, width),
            action=np.asarray(batch.action).reshape(-1),
            reward=np.repeat(np.asarray(batch.reward), robots),
            action_mask=np.stack(batch.action_mask).reshape(batch_size * robots, -1),
            next_action_mask=np.stack(batch.next_action_mask).reshape(batch_size * robots, -1),
            done=np.repeat(np.asarray(batch.done), robots))

    def learn(self, batch, auxiliary_loss=None):
        if auxiliary_loss is not None:
            raise ValueError('Independent primary TD cannot receive auxiliary gradients.')
        loss = self.base.learn(self.local_batch(batch))
        diagnostic = self.base.diagnostic_history[-1]
        diagnostic['primary_td_kind'] = 'independent_shared_team_reward'
        diagnostic['independent_td_loss'] = diagnostic.pop('vdn_td_loss')
        return loss

    def state_dict(self):
        state = self.base.state_dict()
        state['primary_td_kind'] = 'independent_shared_team_reward'
        return state

    def load_state_dict(self, state):
        if state.get('primary_td_kind') != 'independent_shared_team_reward':
            raise ValueError('Checkpoint is not an independent-TD primary.')
        self.base.load_state_dict(state)


class GEVDAblationTrainer(GEVDTrainer):
    def __init__(self, env, config):
        self.ablation = config.get('full_gevd_ablation')
        if self.ablation not in LABELS:
            raise ValueError('Select exactly one supported GEVD ablation.')
        if config.get('training_structure') is not None:
            raise ValueError('Use full_gevd_ablation, not the old B training_structure flag.')
        if float(config['reward']['rho_v']) != .3 or float(config['reward']['rho_g']) != 1.:
            raise ValueError('Common environment reward must stay 0.3/1; ablate replay only.')
        super().__init__(env, config)
        self.config['effective']['implementation_choices']['retrospective_utility'].update(
            primary_td_unchanged=self.ablation != 'no_vdn',
            primary_reward_unchanged=self.ablation == 'no_vdn')
        self.config['effective']['implementation_choices']['full_gevd_ablation'] = dict(
            variant=self.ablation, auxiliary_preserved=True,
            primary_td='independent_shared_team_reward' if self.ablation == 'no_vdn' else 'VDN',
            spectral='S_raw' if self.ablation == 'raw_structure' else 'S',
            training_rho_v=.3, training_rho_g=0. if self.ablation == 'no_gauge' else 1.,
            environment_evaluation_and_selection='common representative S/G, rho_v=.3 rho_g=1')

    @property
    def algorithm_name(self):
        return LABELS[self.ablation]

    def _new_agent(self):
        agent = super()._new_agent()
        if self.ablation == 'no_vdn':
            agent.primary = IndependentTDPrimary(agent.primary)
        return agent

    def _push(self, replay, observations, actions, result, masks):
        channels = result.reward_channels.as_dict()
        if self.ablation == 'raw_structure':
            raw = raw_structural_score(self.env, result.state)
            channels['spectral'] = raw - self._previous_raw_score
            self._previous_raw_score = raw
        elif self.ablation == 'no_gauge':
            channels['gauge'] = 0.
        reward = sum(channels.values())
        self._ablation_training_return += reward
        replay.push(observations, np.asarray(actions, dtype=np.int64), reward,
            result.observations, masks,
            self._terminal_masks(result.action_masks) if result.done else result.action_masks,
            result.done, reward_channels=channels)

    def run_factual_episode(self, training=True, max_steps=None):
        if training:
            self._previous_raw_score = 0.
            self._ablation_training_return = 0.
        episode = super().run_factual_episode(training=training, max_steps=max_steps)
        score = episode['final_structural_score']
        if self.ablation == 'raw_structure':
            score = raw_structural_score(self.env)
        expected = episode['return'] + score - episode['final_structural_score']
        if self.ablation == 'no_gauge':
            expected -= self.env.num_robots - episode['final_component_count']
        if training and not math.isclose(expected, self._ablation_training_return, abs_tol=1e-8):
            raise AssertionError('Ablated primary rewards do not telescope.')
        episode['ablation_training'] = dict(variant=self.ablation,
            structure=score, training_return=expected, common_G=episode['return'],
            replay_modified=training and self.ablation != 'no_vdn', auxiliary_preserved=True)
        return episode

    def load_checkpoint(self, path):
        payload = load_torch_checkpoint(resolve_inside_project(path), map_location='cpu')
        if payload['effective_config'].get('full_gevd_ablation') != self.ablation:
            raise ValueError('GEVD ablation differs from checkpoint.')
        return super().load_checkpoint(path)
