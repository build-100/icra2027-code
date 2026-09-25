"""Intra-only topological information bonus for the C7 coverage QMIX variants.

Potential = alpha * sum_i log(number of spanning trees of robot i's own
unique undirected motion-support graph). Unit weights are intentional: this
rewards extra intra connectivity without importing inter factors or GEVD S.
On current C7 all normalized motion weights are one, so this is also the
weighted scalar information gain. Trees score zero; repeated traversals do
not change the score. This is not an SE(2)/SE(3) covariance measurement.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=16384)
def _score(visited, unique_edges):
    if len(visited) <= 1:
        return 0.0
    # A robot's physically traversed support must be connected.
    index = {v: i for i, v in enumerate(visited)}
    lap = np.zeros((len(visited), len(visited)), dtype=np.float64)
    for u, v in unique_edges:
        i, j = index[u], index[v]
        lap[i, i] += 1.; lap[j, j] += 1.
        lap[i, j] -= 1.; lap[j, i] -= 1.
    sign, value = np.linalg.slogdet(lap[:-1, :-1])
    if sign <= 0:
        raise ValueError('Robot motion-support graph must be connected.')
    if value < -1e-10:
        raise AssertionError('Unweighted connected graph has at least one spanning tree.')
    return max(0., float(value))


def intra_information(env):
    state = env.state
    values = []
    for robot in env.robot_indices:
        visited = tuple(int(v) for v in np.flatnonzero(state.O[robot]))
        edges = tuple(tuple(map(int, env.edge_endpoints[e]))
                      for e in np.flatnonzero(state.M[robot] > 0))
        values.append(env.alpha * _score(visited, edges))
    return {'intra_score': float(sum(values)), 'intra_score_by_robot': values}


def coverage_potential(row, method, intra_weight=1.):
    value = row['coverage'] - row['D'] / (4. * row['distance_bound'])
    if method == 'coverage_first':
        value += .5 * int(row['success'])
    elif method != 'coverage_only':
        raise ValueError(method)
    return float(value + intra_weight * row['intra_score'])
