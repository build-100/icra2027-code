"""Signed, policy-versioned E3 pair supervision in the VDN TD reward units.

No outcome critic, fixed positive credit mass, or action-vs-max margin is used.
The difference of local utilities equals the joint VDN difference when only
one root action changes. Rollouts are reference-conditioned, not Q-star labels.
"""
from __future__ import annotations
import copy
import hashlib
import math
import random
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from gevd.training.event_credit import EventAlignedCreditModule
from gevd.training.learned_event_credit import LearnedEventCreditModule


def network_digest(network):
    h = hashlib.sha256()
    for value in network.state_dict().values():
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class SignedPairAssignment:
    event_id: str
    delay: int
    credit: float
    robot: int
    action: int
    alternative: int


class PairedReturnCreditModule(LearnedEventCreditModule):
    def __init__(self, config, seed, env, reward_scale=1.0):
        EventAlignedCreditModule.__init__(self, config, seed)
        if self.aligner.event_classes != ('E3',) or self.aligner.qualification != 'terminal_success_chain':
            raise ValueError('Stage 1 requires E3 terminal-success-chain roots.')
        self.mode = 'paired_return'
        self.reward_scale = float(reward_scale)
        old = config.get('learned', {})
        settings = config.get('paired_return', {})
        self.max_roots = int(old.get('max_roots_per_episode', 6))
        self.max_alternatives = int(old.get('max_alternatives', 2))
        self.label_interval = int(old.get('label_interval_successes', 4))
        self.max_sim_steps = int(old.get('max_counterfactual_steps', 20000))
        self.capacity = int(old.get('label_capacity', 8000))
        self.max_age_updates = int(settings.get('max_age_updates', 500))
        self.max_uses = int(settings.get('max_uses_per_pair', 32))
        self.cache_factual_returns = bool(settings.get('cache_factual_returns', False))
        self.cached_factual_witnesses = self.cached_factual_steps = 0
        # Optional execution-budget callback, including full candidate validation.
        self.search_budget_remaining = None
        if min(self.max_age_updates, self.max_uses, self.max_roots, self.max_alternatives,
               self.label_interval, self.max_sim_steps, self.capacity) < 1 or self.reward_scale <= 0:
            raise ValueError('Invalid paired-return settings.')
        self.rng = random.Random(seed + 101)
        self.sample_rng = random.Random(seed + 107)
        self.pairs = []
        self.audit = []
        self.counterfactual_steps = self.counterfactual_rollouts = 0
        self.auxiliary_updates = self.labeled_candidates = 0
        self.sign_counts = {'positive': 0, 'negative': 0, 'zero': 0}
        self.feasibility_inversions = 0
        self.skipped_stale_or_reused = 0
        self._agent = None

    def rollout(self, sample, action, cache, env, agent, frozen):
        source = cache[sample.action_time]
        observations, masks = env.restore(source.state)
        phi0 = env.potential()
        joint = list(source.factual_action)
        joint[sample.robot] = int(action)
        total = distance = 0.0
        first = True
        while not env.done:
            if not first:
                joint = list(agent.select_actions(observations, masks, epsilon=0.0, network=frozen))
                t = env.state.time
                if t < len(cache):
                    for robot in range(env.num_robots):
                        if np.array_equal(observations[robot], cache[t].observations[robot]):
                            joint[robot] = cache[t].factual_action[robot]
            result = env.step(joint)
            first = False
            total += result.reward
            distance += result.travel_distance
            self.counterfactual_steps += 1
            observations, masks = result.observations, result.action_masks
        self.counterfactual_rollouts += 1
        expected = env.potential() - phi0 - env.beta * distance
        if not math.isclose(total, expected, rel_tol=1e-9, abs_tol=1e-8):
            raise AssertionError('Paired continuation return does not telescope.')
        return {'G': float(total), 'success': bool(env.success), 'S': env.structural_score(),
                'D_tail': float(distance), 'T': env.state.time, 'coverage': env.coverage_count(),
                'c': env.component_count()}

    def add_episode(self, factual_cache, transitions, *, env, agent):
        if not self.enabled:
            return ()
        self._agent = agent
        self.observed_episode_count += 1
        aligned = [s for s in self.aligner.align_episode(factual_cache, transitions)
                   if int((~s.invalid_action_mask).sum()) > 1]
        if not aligned:
            return ()
        self.qualified_episode_count += 1
        events = {s.event_id for s in aligned}
        self.event_occurrence_count += len(events)
        self.event_ids.update(events)
        if (self.qualified_episode_count - 1) % self.label_interval or self.counterfactual_steps >= self.max_sim_steps:
            return ()
        snapshot = env.snapshot()
        rng_state = agent.rng.getstate()
        frozen = agent.frozen_network()
        version = network_digest(frozen)
        born = int(agent.update_steps)
        accepted = []
        try:
            for sample in self.rng.sample(aligned, min(self.max_roots, len(aligned))):
                alternatives = [int(a) for a in np.flatnonzero(~sample.invalid_action_mask)
                                if a != sample.action][:self.max_alternatives]
                actions = [sample.action, *alternatives]
                simulated_actions = alternatives if self.cache_factual_returns else actions
                bound = len(simulated_actions) * (env.t_max - sample.action_time)
                if self.counterfactual_steps + bound > self.max_sim_steps:
                    continue
                if self.search_budget_remaining is not None and (
                    bound + len(simulated_actions) * env.t_max > self.search_budget_remaining()
                ):
                    continue
                if self.cache_factual_returns:
                    # The reference-conditioned factual witness exactly follows
                    # the recorded trajectory. Reuse its observed return, without
                    # inventing a new observation or charging an unmade step.
                    tail = transitions[sample.action_time:]
                    fact = {'G': float(sum(r.reward for r in tail)),
                            'success': bool(transitions[-1].success),
                            'S': float(env.structural_score(snapshot)),
                            'D_tail': float(sum(r.travel_distance for r in tail)),
                            'T': snapshot.time, 'coverage': env.coverage_count(snapshot),
                            'c': env.component_count(snapshot)}
                    self.cached_factual_witnesses += 1
                    self.cached_factual_steps += len(tail)
                    outcomes = [fact] + [self.rollout(sample, a, factual_cache, env, agent, frozen)
                                         for a in alternatives]
                else:
                    outcomes = [self.rollout(sample, a, factual_cache, env, agent, frozen) for a in actions]
                fact = outcomes[0]
                if not fact['success']:
                    raise AssertionError('Reference-conditioned factual witness was not reproduced.')
                self.labeled_candidates += 1
                for alternative, other in zip(alternatives, outcomes[1:]):
                    difference = fact['G'] - other['G']
                    if other['success']:
                        expected = fact['S'] - other['S'] - env.beta * (fact['D_tail'] - other['D_tail'])
                        if not math.isclose(difference, expected, rel_tol=1e-8, abs_tol=1e-8):
                            raise AssertionError('Successful paired G difference differs from J difference.')
                    # Never manufacture an opposite sign. An inversion between a
                    # feasible witness and a failure is audited and excluded from
                    # auxiliary supervision; ordinary environment TD is unchanged.
                    eligible = other['success'] or difference >= 0.0
                    self.feasibility_inversions += int(not eligible)
                    target = self.reward_scale * difference
                    label = {'observation': np.array(sample.observation, copy=True),
                             'action': int(sample.action), 'alternative': alternative,
                             'target': target, 'born_update': born, 'uses': 0,
                             'policy_sha256': version}
                    if eligible:
                        self.pairs.append(label)
                        self.sign_counts['positive' if target > 1e-10 else 'negative' if target < -1e-10 else 'zero'] += 1
                        accepted.append(SignedPairAssignment(sample.event_id, sample.delay,
                                        target, sample.robot, int(sample.action), alternative))
                        self.assignment_count += 1
                        self.role_counts[sample.role] = self.role_counts.get(sample.role, 0) + 1
                    self.audit.append({'event_id': sample.event_id, 'robot': sample.robot,
                        'action_time': sample.action_time, 'factual_action': int(sample.action),
                        'alternative_action': alternative, 'factual': fact, 'alternative': other,
                        'raw_return_difference': difference, 'scaled_target': target,
                        'eligible': eligible, 'policy_sha256': version, 'born_update': born})
            self.pairs = self.pairs[-self.capacity:]
        finally:
            env.restore(snapshot)
            agent.rng.setstate(rng_state)
        return tuple(accepted)

    @staticmethod
    def pair_loss(network, pairs, device):
        obs = torch.as_tensor(np.stack([p['observation'] for p in pairs]), device=device, dtype=torch.float32)
        q = network(obs)
        rows = torch.arange(len(pairs), device=device)
        actions = torch.as_tensor([p['action'] for p in pairs], device=device)
        alternatives = torch.as_tensor([p['alternative'] for p in pairs], device=device)
        target = torch.as_tensor([p['target'] for p in pairs], device=device, dtype=torch.float32)
        predicted = q[rows, actions] - q[rows, alternatives]
        return nn.functional.smooth_l1_loss(predicted, target), predicted, target

    def auxiliary_loss(self, network, device):
        if not self.enabled or self.loss_weight == 0 or self._agent is None:
            return None
        now = int(self._agent.update_steps)
        fresh = [p for p in self.pairs if now - p['born_update'] <= self.max_age_updates and p['uses'] < self.max_uses]
        self.skipped_stale_or_reused += len(self.pairs) - len(fresh)
        self.pairs = fresh
        if not fresh:
            return None
        batch = self.sample_rng.sample(fresh, min(self.batch_size, len(fresh)))
        loss, predicted, target = self.pair_loss(network, batch, device)
        for pair in batch:
            pair['uses'] += 1
        self.auxiliary_updates += 1
        self.last_diagnostics = {'sample_count': len(batch), 'raw_loss': float(loss.detach()),
            'weighted_loss': float(self.loss_weight * loss.detach()),
            'mean_absolute_pair_error': float((predicted - target).abs().mean().detach())}
        return self.loss_weight * loss

    def diagnostics(self):
        result = EventAlignedCreditModule.diagnostics(self)
        result.update(mode=self.mode, counterfactual_steps=self.counterfactual_steps,
            counterfactual_rollouts=self.counterfactual_rollouts, labeled_candidates=self.labeled_candidates,
            critic_updates=0, auxiliary_updates=self.auxiliary_updates, fixed_total_credit=False,
            reward_scale=self.reward_scale, max_age_updates=self.max_age_updates,
            max_uses_per_pair=self.max_uses, sign_counts=dict(self.sign_counts),
            feasibility_inversions=self.feasibility_inversions, label_pair_count=len(self.audit),
            pair_replay_size=len(self.pairs), counterfactual_continuation='factual_local_history_else_frozen_greedy',
            contribution_claim='signed_return_difference_under_recorded_reference_continuation')
        result.update(cache_factual_returns=self.cache_factual_returns,
                      cached_factual_witnesses=self.cached_factual_witnesses,
                      cached_factual_steps=self.cached_factual_steps)
        return result

    def state_dict(self):
        result = EventAlignedCreditModule.state_dict(self)
        result['paired_state'] = {key: copy.deepcopy(value) for key, value in self.__dict__.items()
                                  if key in ('pairs', 'audit', 'counterfactual_steps', 'counterfactual_rollouts',
                                             'auxiliary_updates', 'labeled_candidates', 'sign_counts',
                                             'feasibility_inversions', 'skipped_stale_or_reused',
                                             'cached_factual_witnesses', 'cached_factual_steps')}
        result['paired_state'].update(rng_state=self.rng.getstate(), sample_rng_state=self.sample_rng.getstate())
        return result

    def load_state_dict(self, state):
        if 'paired_state' not in state:
            raise ValueError('A legacy credit checkpoint cannot initialize paired-return credit.')
        EventAlignedCreditModule.load_state_dict(self, state)
        values = copy.deepcopy(state['paired_state'])
        self.rng.setstate(values.pop('rng_state'))
        self.sample_rng.setstate(values.pop('sample_rng_state'))
        self.__dict__.update(values)
