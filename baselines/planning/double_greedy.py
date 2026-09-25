"""Ordered randomized double greedy on executable GEVD loop-insertion sets.

The oracle is the native four-term return, not the original CGE Laplacian
surrogate. Horizon truncation and feasibility destroy any inherited USM bound.
"""
import math
import random

from .planners import CoordinatedPlanner, PlanningBudget


def ordered_double_greedy(ground, oracle, rng, audit=None):
    """Maintain X <= Y; dynamically order by f(X+u)-f(X).

    Match the public CGE implementation: add with probability a/(a+b),
    with probability one when a=b=0. No lazy bounds are assumed.
    `audit` also survives a budget exception from the oracle.
    """
    ground = tuple(ground)
    if len(set(ground)) != len(ground):
        raise ValueError('Ground-set elements must be unique')
    x, y, remaining = frozenset(), frozenset(ground), list(ground)
    audit = {} if audit is None else audit
    audit.update(completed=False, decisions=[], x=[], y=list(ground))

    def value(subset):
        result = float(oracle(subset))
        if not math.isfinite(result):
            raise ValueError('Double-greedy oracle must be finite for every subset')
        return result

    fx, fy = value(x), value(y)
    while remaining:
        additions = [(u, value(x | {u})) for u in remaining]
        # Stable first-element tie breaking, independent of the RNG.
        u, fadd = max(additions, key=lambda pair: pair[1])
        fremove = value(y - {u})
        a, b = max(0., fadd - fx), max(0., fremove - fy)
        probability = a / (a + b) if a + b else 1.
        draw = rng.random()
        accepted = draw < probability
        audit['decisions'].append(dict(element=u, a=a, b=b, probability=probability,
            draw=draw, accepted=accepted, x_before=list(sorted(x)), y_before=list(sorted(y))))
        if accepted:
            x, fx = x | {u}, fadd
        else:
            y, fy = y - {u}, fremove
        remaining.remove(u)
        assert x <= y and y - x == frozenset(remaining)
        audit.update(x=list(sorted(x)), y=list(sorted(y)))
    assert x == y
    audit.update(completed=True, final_value=fx)
    return x, fx


class CGEDoubleGreedyPlanner(CoordinatedPlanner):
    """Coverage/fusion shared with sGre; a fixed ground set per base route."""
    def __init__(self, env, budget=25000, seed=906300, ground_limit=32):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError('budget must be a nonnegative integer')
        super().__init__(env, 'cge_dgre_order', budget, seed)
        if ground_limit is not None and (isinstance(ground_limit, bool)
                or not isinstance(ground_limit, int) or ground_limit < 1):
            raise ValueError('ground_limit must be a positive integer or None')
        self.ground_limit = ground_limit
        self.rng = random.Random(seed)
        env.reset()
        self.initial = dict(S=env.structural_score(), coverage=env.coverage_count(),
                            c=env.component_count())
        self.double_greedy_runs = []

    def evaluate(self, routes):
        row = super().evaluate(routes)
        row['G'] = (row['S'] - self.initial['S']
                    + self.env.rho_V * (row['coverage'] - self.initial['coverage'])
                    + self.env.rho_g * (self.initial['c'] - row['c'])
                    - self.env.beta * row['D'])
        if not math.isclose(row['G'], row['return'], rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError('Endpoint G differs from summed native rewards')
        self.history[-1].update(G=row['G'], best_G=self.best['G'] if self.best else None)
        return row

    def ground_set(self, base):
        # Same two-step, synchronous excursion neighborhood as the old planner.
        # Locations refer to the BASE path, never to already shifted indices.
        return [(t, i, v) for t in range(base['T'])
                for i, route in enumerate(base['routes'])
                for v in sorted(self.graph.neighbors(route[t]))]

    def compose(self, base, ground, subset):
        """Canonical composition makes the oracle independent of decision order.

        Each insertion returns ALL robots to their base anchors. Multiple
        insertions at one base time are executed sequentially in ground order.
        Full routes are legal; evaluate applies first-success/T_max termination.
        """
        by_time = {}
        for index in sorted(subset):
            t, i, v = ground[index]
            by_time.setdefault(t, []).append((i, v))
        routes = [[r[0]] for r in base['routes']]
        for t in range(base['T']):
            for i, v in by_time.get(t, []):
                for j, route in enumerate(routes):
                    anchor = base['routes'][j][t]
                    excursion = v if i == j else self.padding(anchor)
                    route.extend([excursion, anchor])
            for j, route in enumerate(routes):
                route.append(base['routes'][j][t + 1])
        return routes

    def improve(self, base):
        ground = self.ground_set(base)
        cache = {frozenset(): base}
        audit = dict(base_routes=base['routes'], ground=[list(u) for u in ground],
                     oracle_requests=0, cache_hits=0, replayed_subsets=0,
                     completed=False, decisions=[], ground_limit=self.ground_limit,
                     phase='singleton_screening')
        self.double_greedy_runs.append(audit)

        def oracle(subset):
            audit['oracle_requests'] += 1
            if subset in cache:
                audit['cache_hits'] += 1
            else:
                cache[subset] = self.evaluate(self.compose(base, ground, subset))
                audit['replayed_subsets'] += 1
            return cache[subset]['G']

        try:
            active = list(range(len(ground)))
            if self.ground_limit is not None and len(active) > self.ground_limit:
                # Fixed-budget adaptation, NOT submodularity-based safe pruning.
                # Every singleton query is charged and eligible for common retention.
                scores = [(u, oracle(frozenset({u}))) for u in active]
                ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
                active = sorted(u for u, _ in ranked[:self.ground_limit])
                audit['singleton_scores'] = scores
            audit.update(active_ground=active, phase='double_greedy')
            selected, _ = ordered_double_greedy(active, oracle, self.rng, audit)
        except PlanningBudget:
            audit['stop_reason'] = 'simulator_budget'
            raise
        audit['selected_row'] = cache[selected]
        audit['stop_reason'] = 'all_elements_processed'
        return cache[selected]

    def run(self):
        result = super().run()
        result.update(algorithm='CGE-dGre+order (adapted)', seed=self.seed,
            optimizer_updates=0, training_steps=0, initial=self.initial,
            rho_v=self.env.rho_V, rho_g=self.env.rho_g, beta=self.env.beta,
            ground_limit=self.ground_limit,
            double_greedy_runs=self.double_greedy_runs,
            selection='best fully replayed feasible G among all queried candidates',
            approximation_guarantee=False,
            oracle='native G after first-success or horizon truncation')
        return result
