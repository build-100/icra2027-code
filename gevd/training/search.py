"""Budgeted GEVD search with verified complete-path admission.

The collector accepts actual completed search trajectories, never evaluation
episodes. Replay verification is charged before an incumbent becomes available.
This is online candidate retention, not a post-training path optimizer.
"""
import copy
import hashlib
import json
import math
from collections import Counter
from contextlib import contextmanager
import numpy as np
from gevd.training.base import VDNTrainer
from gevd.environments.multi_robot import GEVDMultiRobotEnv
from gevd.training.paired_return_credit import PairedReturnCreditModule

ENV_STEP = GEVDMultiRobotEnv.step
SEARCH_PHASES = ('factual', 'mcbr', 'credit', 'planner', 'candidate_validation')


def state_digest(state):
    h = hashlib.sha256()
    for x in (state.M, state.O, state.owners, state.positions):
        h.update(x.tobytes())
    h.update(str(state.time).encode())
    return h.hexdigest()


def route_digest(routes):
    return hashlib.sha256(json.dumps(routes, separators=(',', ':')).encode()).hexdigest()


def prefix_routes(env, cache, t):
    return [[env.node_labels[int(cache[k].state.positions[i])] for k in range(t + 1)]
            for i in range(env.num_robots)]


class SearchLedger:
    def __init__(self, budget, out=None):
        self.budget = int(budget)
        self.out = out
        self.counts = Counter()
        self.phase, self.context = 'idle', None
        self.best = None
        self.events, self.rollouts = [], []
        self.unique = set()
        self.queries, self.skipped = Counter(), Counter()
        self.episode = 0

    @property
    def total(self):
        return sum(self.counts[k] for k in SEARCH_PHASES)

    @property
    def remaining(self):
        return self.budget - self.total

    def emit(self, name, row):
        if self.out:
            with (self.out / name).open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    @contextmanager
    def mode(self, phase, context=None):
        old = self.phase, self.context
        self.phase, self.context = phase, context
        try:
            yield
        finally:
            self.phase, self.context = old

    @contextmanager
    def installed(self):
        old = GEVDMultiRobotEnv.step
        GEVDMultiRobotEnv.step = lambda env, actions: self.step(env, actions)
        try:
            yield
        finally:
            GEVDMultiRobotEnv.step = old

    def trajectory(self, source, routes, **extra):
        return dict(source=source, routes=copy.deepcopy(routes), start_C=self.total,
                    episode=self.episode, **extra)

    def step(self, env, actions):
        if self.phase in SEARCH_PHASES and self.remaining <= 0:
            raise RuntimeError('Total search simulation budget exceeded')
        value = ENV_STEP(env, actions)
        self.counts[self.phase] += 1
        if self.context is not None:
            for i, v in enumerate(env.current_labels):
                self.context['routes'][i].append(v)
            if env.done:
                self.complete(env)
        return value

    def complete(self, env):
        ctx = self.context
        assert ctx['source'] in ('factual', 'mcbr', 'credit_alternative', 'planner')
        state, routes = env.snapshot(), ctx['routes']
        assert all(len(r) == state.time + 1 for r in routes)
        distance = float(np.sum(state.M * env.edge_distances[None, :]))
        score = float(env.structural_score())
        row = {k: v for k, v in ctx.items() if k != 'routes'}
        row.update(discovered_C=self.total, calls=self.total-ctx['start_C'],
                   success=bool(env.success), S=score, D=distance, J=score-env.beta*distance,
                   T=state.time, coverage=env.coverage_count(), c=env.component_count(),
                   routes=copy.deepcopy(routes), path_sha256=route_digest(routes),
                   state_sha256=state_digest(state), overlap=int((state.O.sum(0) >= 2).sum()),
                   best_before=self.best['J'] if self.best else None, admitted=False)
        key = ctx.get('root_key')
        if key:
            row['repeat_query'] = self.queries[(ctx['source'], key)] > 0
            self.queries[(ctx['source'], key)] += 1
        if row['success']:
            row['new_unique_feasible'] = row['path_sha256'] not in self.unique
            self.unique.add(row['path_sha256'])
            if self.best is None or row['J'] > self.best['J'] + 1e-12:
                if self.remaining >= row['T']:
                    # Validate the entire reconstructed prefix and suffix from
                    # reset; scoring/owner assignment must match the source.
                    with self.mode('candidate_validation'):
                        env.reset()
                        try:
                            for t in range(row['T']):
                                assert env.current_labels == tuple(r[t] for r in routes)
                                env.step([env.prior.action_for_neighbor(r[t], r[t+1]) for r in routes])
                            assert env.success and state_digest(env.state) == row['state_sha256']
                            assert abs(env.structural_score() - row['S']) < 1e-9
                        finally:
                            env.restore(state)
                    row.update(admitted=True, C=self.total)
                    self.best = copy.deepcopy(row)
                else:
                    self.skipped['insufficient_budget_for_candidate_validation'] += 1
        row['C'] = self.total
        row['best_J'] = self.best['J'] if self.best else None
        if row['success']:
            self.events.append(copy.deepcopy(row))
            self.emit('candidate_events.jsonl', row)
        if ctx['source'] != 'factual':
            self.rollouts.append(row)
            self.emit('rollouts.jsonl', row)
        ctx['completion'] = row


def query_key(state, actions):
    return hashlib.sha256((state_digest(state) + repr(tuple(actions))).encode()).hexdigest()


class BudgetedSearchTrainer(VDNTrainer):
    def bind_search(self, ledger):
        self.ledger = ledger
        self.update_basis = self.config.get('search', {}).get('update_basis', 'factual')
        if self.update_basis not in ('factual','factual_plus_mcbr'):
            raise ValueError('Unknown search update_basis')
        self._scheduled_branch_count = int(self.branch_transition_count)
        self._fractional_update_credit = 0.0
        self.update_schedule = Counter()
        module = self.event_credit
        if not isinstance(module, PairedReturnCreditModule):
            return
        if not module.cache_factual_returns:
            raise ValueError('Budgeted production search requires cached factual returns.')
        module.search_budget_remaining = lambda: ledger.remaining
        original = module.rollout

        def rollout(sample, action, cache, env, agent, frozen):
            assert int(action) != sample.action
            t = sample.action_time
            joint = list(cache[t].factual_action)
            joint[sample.robot] = int(action)
            ctx = ledger.trajectory('credit_alternative', prefix_routes(env, cache, t),
                                    root_key=query_key(cache[t].state, joint),
                                    root_time=t, changed_robot=int(sample.robot), root_actions=joint)
            with ledger.mode('credit', ctx):
                result = original(sample, action, cache, env, agent, frozen)
            assert 'completion' in ctx
            return result
        module.rollout = rollout

    def _updates_if_ready(self, factual_steps=None):
        if self.update_basis == 'factual':
            return super()._updates_if_ready(factual_steps)
        if factual_steps is None or int(factual_steps) < 0:
            raise ValueError('Nonnegative factual_steps required for replay-step scheduling')
        branch_steps = int(self.branch_transition_count)-self._scheduled_branch_count
        if branch_steps < 0:
            raise ValueError('Branch counter moved backwards')
        self._scheduled_branch_count = int(self.branch_transition_count)
        eligible = int(factual_steps)+branch_steps
        if len(self.factual_replay) < int(self.vdn_config['min_factual_replay']):
            self.update_schedule['warmup_replay_steps'] += eligible
            return []
        # Credit-label rollouts, exams and candidate validation are not TD
        # replay transitions and therefore receive no optimizer quota.
        rate = float(self.vdn_config['updates_per_factual_step'])
        if not math.isfinite(rate) or rate < 0:
            raise ValueError('Update rate must be finite and nonnegative')
        self._fractional_update_credit += rate*eligible
        quota = math.floor(self._fractional_update_credit+1e-12)
        self._fractional_update_credit -= quota
        self.update_schedule.update(eligible_factual_steps=int(factual_steps),
            eligible_mcbr_steps=branch_steps,scheduled_updates=quota)
        losses = self._learn_updates(quota)
        self.ledger.emit('update_schedule.jsonl',dict(C=self.ledger.total,
            factual_steps=int(factual_steps),mcbr_steps=branch_steps,quota=quota,
            total_optimizer_updates=self.agent.update_steps,
            fractional_credit=self._fractional_update_credit))
        return losses

    def _run_branch(self, cache, root, frozen_network, frozen_epsilon, digest):
        t, robot, alt = root
        if self.ledger.remaining < (self.env.t_max-t) + self.env.t_max:
            self.ledger.skipped['mcbr_budget_reservation'] += 1
            return dict(root=root, records=[], success=False, budget_skipped=True)
        joint = list(cache[t].factual_action)
        joint[robot] = alt
        ctx = self.ledger.trajectory('mcbr', prefix_routes(self.env, cache, t),
                root_key=query_key(cache[t].state, joint), root_time=t, changed_robot=robot,
                root_actions=joint, epsilon=frozen_epsilon, policy_sha256=digest)
        with self.ledger.mode('mcbr', ctx):
            result = super()._run_branch(cache, root, frozen_network, frozen_epsilon, digest)
        assert 'completion' in ctx
        return result


def run_budgeted_config(config, budget):
    """Main test3 CLI with native return and read-only current-policy evaluation."""
    from gevd.training.runner import run_budgeted_main
    return run_budgeted_main(config, budget)
