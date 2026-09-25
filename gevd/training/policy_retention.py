"""Optional self-generated successful-witness regularization, not GCR reward.

Selection may use the factual team outcome during training. The Q network sees
only the same local observations/masks as deployment. This is an explicit
algorithm revision; it is not E3 credit or a uniquely causal contribution.
"""
import copy
import math
import random

import numpy as np
import torch


class SuccessfulWitnessRetention:
    def __init__(self, config=None, seed=0):
        self.config = dict(config or {})
        self.enabled = self.config.get('enabled', False)
        if not isinstance(self.enabled, bool):
            raise TypeError('policy_retention.enabled must be boolean')
        self.weight = float(self.config.get('loss_weight', 1.0))
        self.margin = float(self.config.get('action_margin', 0.1))
        self.batch_size = int(self.config.get('batch_size', 32))
        if self.batch_size < 1 or not all(math.isfinite(x) and x > 0 for x in (self.weight, self.margin)):
            raise ValueError('Retention margin, weight and batch size must be positive')
        self.rng = random.Random(seed + 211)
        self.witness = None
        self.replacements = 0
        self.loss_calls = 0
        self.last_loss = 0.0

    def consider(self, episode):
        if not self.enabled or not episode['success'] or not episode['terminal']:
            return False
        score = float(episode['return'])
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite witness return')
        routes = copy.deepcopy(episode['routes'])
        if self.witness is not None:
            previous = self.witness['return']
            better = score > previous + 1e-12
            tied_better = math.isclose(score, previous, abs_tol=1e-12) and repr(routes) < repr(self.witness['routes'])
            if not (better or tied_better):
                return False
        cache = episode['pre_action_cache']
        if not cache:
            raise ValueError('Only a factual training episode may supply a witness')
        obs = np.concatenate([x.observations for x in cache]).copy()
        masks = np.concatenate([x.action_masks for x in cache]).copy()
        actions = np.concatenate([x.factual_action for x in cache]).astype(np.int64)
        if masks[np.arange(len(actions)), actions].any():
            raise ValueError('Witness contains an invalid action')
        usable = (~masks).sum(axis=1) > 1
        self.witness = {'return': score, 'routes': routes,
            'observations': obs[usable], 'masks': masks[usable], 'actions': actions[usable],
            'joint_steps': int(episode['joint_steps']), 'source': 'successful_factual_only'}
        self.replacements += 1
        return True

    def auxiliary_loss(self, network, device):
        self.last_loss = 0.0
        if not self.enabled or self.witness is None or not len(self.witness['actions']):
            return None
        w = self.witness
        indices = self.rng.sample(range(len(w['actions'])), min(self.batch_size, len(w['actions'])))
        obs = torch.as_tensor(w['observations'][indices], dtype=torch.float32, device=device)
        masks = torch.as_tensor(w['masks'][indices], dtype=torch.bool, device=device)
        actions = torch.as_tensor(w['actions'][indices], dtype=torch.long, device=device)
        q = network(obs)
        row = torch.arange(len(indices), device=device)
        alternatives = masks.clone()
        alternatives[row, actions] = True
        best_alternative = q.masked_fill(alternatives, -torch.inf).max(dim=1).values
        loss = self.weight * torch.relu(self.margin - (q[row, actions] - best_alternative)).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite successful-witness loss')
        self.last_loss = float(loss.detach())
        self.loss_calls += 1
        return loss

    def diagnostics(self):
        return {'enabled': self.enabled, 'witness_available': self.witness is not None,
            'replacements': self.replacements, 'loss_calls': self.loss_calls, 'last_loss': self.last_loss,
            'witness_return': self.witness['return'] if self.witness else None,
            'local_examples': len(self.witness['actions']) if self.witness else 0,
            'role': 'self_generated_witness_regularizer_not_environment_reward'}

    def state_dict(self):
        return {'config': copy.deepcopy(self.config), 'witness': copy.deepcopy(self.witness),
            'rng': self.rng.getstate(), 'replacements': self.replacements,
            'loss_calls': self.loss_calls, 'last_loss': self.last_loss}

    def load_state_dict(self, state):
        if state['config'] != self.config:
            raise ValueError('Retention config differs from checkpoint')
        self.witness = copy.deepcopy(state['witness'])
        self.rng.setstate(state['rng'])
        for name in ('replacements', 'loss_calls', 'last_loss'):
            setattr(self, name, state[name])
