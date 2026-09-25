"""Exact shared budget/candidate code; no learner or credit imports."""
import copy,hashlib,json,math
from collections import Counter
from contextlib import contextmanager
import numpy as np
from gevd.environments.multi_robot import GEVDMultiRobotEnv
ENV_STEP=GEVDMultiRobotEnv.step
SEARCH_PHASES=('factual','mcbr','credit','planner','candidate_validation')

def state_digest(state):
    h = hashlib.sha256()
    for x in (state.M, state.O, state.owners, state.positions):
        h.update(x.tobytes())
    h.update(str(state.time).encode())
    return h.hexdigest()

def route_digest(routes):
    return hashlib.sha256(json.dumps(routes, separators=(',', ':')).encode()).hexdigest()

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
