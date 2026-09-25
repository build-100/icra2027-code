"""GEVD: factual VDN plus an independent retrospective utility branch."""
from gevd.training.search import BudgetedSearchTrainer
from gevd.training.retrospective_utility import RetrospectiveUtilityAgent, resolve_settings


class GEVDTrainer(BudgetedSearchTrainer):
    def __init__(self, env, config):
        settings = resolve_settings(config.get('retrospective_utility', {}))
        if settings['max_trace_steps'] < env.t_max:
            raise ValueError('max_trace_steps must cover this environment horizon.')
        if config.get('event_aligned', {}).get('enabled', False):
            raise ValueError('GEVD requires the old event/paired-search module disabled.')
        if (config['mcbr']['branch_roots_per_episode'] or config['mcbr']['branch_fraction']
                or config.get('policy_retention', {}).get('enabled', False)):
            raise ValueError('GEVD uses factual replay without MCBR or witness auxiliary loss.')
        super().__init__(env, config)
        self.config['retrospective_utility'] = settings
        self.config['effective']['implementation_choices']['retrospective_utility'] = dict(
            separate_encoders=True, separate_optimizers=True, zero_initial_output=True,
            bound='abs(Q_prime) <= 0.1 * abs(Q), before execution_lambda',
            target_policy='current masked greedy Q + execution_lambda * Q_prime',
            primary_td_unchanged=True, primary_reward_unchanged=True,
            reward_scale=self.agent.td_reward_scale, joint_importance_sampling=True,
            factual_segments_only=True, includes_failed_and_non_fusing_episodes=True)

    @property
    def algorithm_name(self):
        return 'GEVD'

    def _new_agent(self):
        return RetrospectiveUtilityAgent(super()._new_agent(),
            self.config.get('retrospective_utility', {}))

    def run_factual_episode(self, training=True, max_steps=None):
        if not training:
            return super().run_factual_episode(training=False, max_steps=max_steps)
        self.agent.capture = []
        try:
            episode = super().run_factual_episode(training=True, max_steps=max_steps)
            records = self.agent.capture
        finally:
            self.agent.capture = None
        events = self.agent.add_episode(episode['pre_action_cache'], episode['transitions'],
            [record['log_mu'] for record in records])
        episode['retrospective_utility'] = dict(events=events,
            max_qprime_to_q_ratio=max((r['max_ratio'] for r in records), default=0.0),
            execution_lambda=self.agent.settings['execution_lambda'],
            greedy_action_changes=sum(r['greedy_changes'] for r in records),
            local_decisions=len(records) * self.env.num_robots)
        return episode
