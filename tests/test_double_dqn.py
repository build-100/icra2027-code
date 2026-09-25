"""Disagreeing online/target rankings detect accidental ordinary-DQN targets."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from gevd.models.dqn import DQNAgent


def agent(double=True):
    a = DQNAgent(3, 3, 4, .001, .5, 10, 10., 7,
                 double_dqn=double, td_reward_scale=.1)
    # Distinct legal online winners for each robot; invalid action 2 is tempting.
    a.online = torch.nn.Linear(3, 3, bias=False)
    a.target = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        a.online.weight.copy_(torch.tensor([[9., 1., 0.], [1., 9., 0.], [99., 99., 0.]]))
        a.target.weight.copy_(torch.tensor([[2., 7., 0.], [8., 3., 0.], [100., 100., 0.]]))
    return a


def test_double_selects_online_evaluates_target_masks_and_sums_team_once():
    states = np.array([[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]], np.float32)
    masks = np.array([[[0, 0, 1], [0, 0, 1]], [[1, 1, 1], [1, 1, 1]]], bool)
    a = agent()
    result = a.compute_td_targets([10., 20.], states, masks, [False, True])
    # DDQN: .1*10 + .5*(2+3), terminal: .1*20. DQN instead uses 8+7.
    np.testing.assert_allclose(result.numpy(), [3.5, 2.])
    assert not result.requires_grad
    np.testing.assert_allclose(agent(False).compute_td_targets(
        [10., 20.], states, masks, [False, True]).numpy(), [8.5, 2.])


def test_checkpoint_rule_mismatch_rejected_before_loading_weights():
    ordinary = DQNAgent(3, 3, 4, .001, 1., 10, 10., 7)
    double = DQNAgent(3, 3, 4, .001, 1., 10, 10., 7, double_dqn=True)
    before = copy.deepcopy(double.online.state_dict())
    old = ordinary.state_dict()
    old.pop('double_dqn')  # Previous run's legacy checkpoint has no rule field.
    with pytest.raises(ValueError, match='target rule'):
        double.load_state_dict(old)
    assert all(torch.equal(v, double.online.state_dict()[k]) for k, v in before.items())
    double.load_state_dict(double.state_dict())


def test_invalid_nonterminal_actions_rejected():
    with pytest.raises(ValueError, match='no valid action'):
        agent().compute_td_targets([0.], np.zeros((1, 2, 3), np.float32),
                                   np.ones((1, 2, 3), bool), [False])
