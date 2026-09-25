"""Scientific invariants of the new bounded, independent GEVD branch."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
from gevd.training.gevd import GEVDTrainer
from gevd.training.search import BudgetedSearchTrainer, SearchLedger
from gevd.training.base import _network_digest
from gevd.evaluation.replay import preserved_readonly, training_signature, object_digest
from gevd.training.retrospective_utility import (ABSOLUTE_QPRIME_RATIO, bounded_utility,
    collect_auxiliary_episode, masked_policy, truncated_retrace)


@pytest.fixture
def cfg(tmp_path):
    config = yaml.safe_load((ROOT / 'configs/simulation/comparison/map8/N3/gevd.yaml').read_text(encoding='utf-8'))
    config['vdn'].update(device='cpu', hidden_dim=16, batch_size=2, min_factual_replay=2,
                         target_update_interval=2, cpu_threads=1)
    config['retrospective_utility']['batch_size'] = 2
    config['output']['directory'] = str(tmp_path)
    return config


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cap_is_per_action_sign_safe_zero_safe_and_independent_of_lambda(dtype):
    q = torch.tensor([-100., -1., -1e-12, 0., 1e-12, 1., 100.], dtype=dtype, requires_grad=True)
    z = torch.tensor([1e30, -1e30, 1., 3., -2., .5, -1.], dtype=dtype, requires_grad=True)
    qp = bounded_utility(q, z)
    assert torch.all(qp.abs() <= .1 * q.abs())
    assert qp[3] == 0
    for execution_lambda in (0., .1, .8):
        correction = execution_lambda * qp
        assert torch.all(correction.abs() <= execution_lambda * .1 * q.abs() + torch.finfo(dtype).eps)
    qp.sum().backward()
    assert q.grad is None
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_retrace_on_policy_terminal_and_truncated_bootstrap():
    r = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    q = torch.tensor([[.2, .3, .4], [.2, .3, .4]])
    v = torch.tensor([[.3, .4, 99.], [.3, .4, .7]])
    done = torch.tensor([[False, False, True], [False, False, False]])
    target = truncated_retrace(r, q, v, torch.ones_like(r), done, torch.ones_like(done))
    torch.testing.assert_close(target, torch.tensor([[1., 1., 1.], [1.7, 1.7, 1.7]]))
    c = torch.ones_like(r)
    c[:, 1] = 0  # A teammate's off-policy action cuts later credit at the preceding step.
    target = truncated_retrace(r, q, v, c, done, torch.ones_like(done))
    torch.testing.assert_close(target[:, 0], torch.tensor([.3, .3]))
    assert not target.requires_grad


def test_masked_policy_behavior_likelihood_and_ties():
    q = torch.tensor([[2., 2., 900.], [-3., -2., 800.]], dtype=torch.float64)
    mask = torch.tensor([[False, False, True], [False, False, True]])
    policy = masked_policy(q, mask, .2)
    torch.testing.assert_close(policy, torch.tensor([[.9, .1, 0.], [.1, .9, 0.]], dtype=torch.float64))
    assert float(policy[0, 0] * policy[1, 1]) == pytest.approx(.81)
    assert not masked_policy(q, torch.ones_like(mask), .2).any()


def synthetic_history(initial=False, simultaneous=False, duplicate=False, event_class='E3'):
    # Region 2: robot 0 first enters on move 0, robot 1 arrives on move 2.
    visits = np.zeros((3, 4), bool)
    visits[0, 0] = visits[1, 1] = visits[2, 3] = True
    if initial:
        visits[0, 2] = True
    cache, transitions = [], []
    for t in range(3):
        cache.append(SimpleNamespace(absolute_time=t, state=SimpleNamespace(O=visits.copy()),
            factual_action=(0, 0, 0), observations=np.ones((3, 2), np.float32) * t,
            action_masks=np.zeros((3, 2), bool)))
        if t == 0 and not simultaneous:
            visits[0, 2] = True
        if t == 2:
            visits[0, 2] = visits[1, 2] = True
        event = SimpleNamespace(factor_type='closure', event_class=event_class,
            robot_pair=(0, 1), region_index=2)
        events = ([event, event] if duplicate else [event]) if t == 2 else []
        transitions.append(SimpleNamespace(state=SimpleNamespace(O=visits.copy()),
            observations=np.ones((3, 2), np.float32) * (t+1), action_masks=np.zeros((3, 2), bool),
            done=t == 2, success=False, events=events))
    return cache, transitions


def test_delayed_reward_recorded_at_closure_once_for_earlier_robot_even_in_failed_episode():
    cache, transitions = synthetic_history(duplicate=True)
    episode = collect_auxiliary_episode(cache, transitions, [0., 0., 0.])
    assert episode.event_roots == ((0, 0, 3),)
    assert episode.rewards[2, 0] == .5
    assert episode.rewards.sum() == .5
    assert not episode.rewards[:2].any()
    assert episode.invalid_masks[-1].all()


@pytest.mark.parametrize('options', [dict(initial=True), dict(simultaneous=True), dict(event_class='E2')])
def test_initial_simultaneous_and_e2_are_not_retrospective_events(options):
    cache, transitions = synthetic_history(**options)
    episode = collect_auxiliary_episode(cache, transitions, [0., 0., 0.])
    assert not episode.event_roots and not episode.rewards.any()


def test_initial_primary_rng_and_zero_auxiliary_match_existing_backbone(cfg):
    full = GEVDTrainer.from_config(cfg)
    random_state = torch.get_rng_state().clone()
    base_cfg = copy.deepcopy(cfg)
    base_cfg.pop('retrospective_utility')
    base = BudgetedSearchTrainer.from_config(base_cfg)
    assert torch.equal(torch.get_rng_state(), random_state)
    assert _network_digest(full.agent.online) == _network_digest(base.agent.online)
    assert not ({p.data_ptr() for p in full.agent.online.parameters()} &
                {p.data_ptr() for p in full.agent.auxiliary.parameters()})
    observations = full.env.local_observations()
    masks = full.env.action_masks()
    for eps in (0., .1, 1.):
        np.testing.assert_array_equal(full.agent.select_actions(observations, masks, eps),
                                      base.agent.select_actions(observations, masks, eps))


def test_training_primary_update_unchanged_auxiliary_gradients_isolated_and_restore(cfg, tmp_path):
    full = GEVDTrainer.from_config(cfg)
    episode = full.run_factual_episode()
    assert full.agent.auxiliary_episodes == 1
    assert len(full.agent.auxiliary_replay[0].behavior_log_prob) == episode['joint_steps']
    assert np.isfinite(full.agent.auxiliary_replay[0].behavior_log_prob).all()
    # Guarantee a nonzero auxiliary training target without relying on stochastic success.
    full.agent.auxiliary_replay[0].rewards[:] = .5
    primary_before = _network_digest(full.agent.online)
    primary_optimizer_before = object_digest(full.agent.optimizer.state_dict())
    aux_before = _network_digest(full.agent.auxiliary)
    full.agent.learn_auxiliary()
    assert _network_digest(full.agent.auxiliary) != aux_before
    assert _network_digest(full.agent.online) == primary_before
    assert object_digest(full.agent.optimizer.state_dict()) == primary_optimizer_before
    assert all(p.grad is None for p in full.agent.online.parameters())
    base_cfg = copy.deepcopy(cfg)
    base_cfg.pop('retrospective_utility')
    base = BudgetedSearchTrainer.from_config(base_cfg)
    base.agent.load_state_dict(full.agent.primary.state_dict())
    batch = full.factual_replay.sample(2)
    base.agent.learn(batch)
    full.agent.learn(batch)
    assert _network_digest(full.agent.online) == _network_digest(base.agent.online)
    assert object_digest(full.agent.optimizer.state_dict()) == object_digest(base.agent.optimizer.state_dict())
    np.testing.assert_array_equal(full.agent.last_td_targets, base.agent.last_td_targets)
    observations = torch.tensor(episode['pre_action_cache'][0].observations)
    with torch.no_grad():
        q, qp, combined = full.agent.execution_values(observations)
    assert torch.all(qp.abs() <= .1 * q.abs())
    assert qp.abs().max() > 0
    full.agent.diagnostic_history.clear()  # The production logger flushes these before checkpointing.
    path = full.save_checkpoint(tmp_path / 'full.pt')
    restored = GEVDTrainer.from_config(cfg)
    restored.load_checkpoint(path)
    assert training_signature(full) == training_signature(restored)
    # Resume must reproduce replay sampling, targets, auxiliary and primary updates.
    full._learn_updates(2)
    restored._learn_updates(2)
    assert training_signature(full) == training_signature(restored)
    with pytest.raises(ValueError, match='Retrospective utility'):
        base.load_checkpoint(path)
    with pytest.raises(ValueError, match='auxiliary branch'):
        base.agent.load_state_dict(full.agent.state_dict())


def test_evaluation_preserves_both_branches_and_replay_and_budget(cfg):
    full = GEVDTrainer.from_config(cfg)
    full.run_factual_episode()
    full.agent.auxiliary_replay[0].rewards[:] = .5
    full._learn_updates(1)
    ledger = SearchLedger(60)
    full.bind_search(ledger)
    before = training_signature(full)
    with ledger.installed():
        with preserved_readonly(full, ledger):
            with ledger.mode('evaluation'):
                full.run_factual_episode(training=False)
    assert training_signature(full) == before
    assert ledger.total == 0 and ledger.counts['evaluation'] > 0
    assert not ledger.best


def test_small_budget_pipeline_logs_auxiliary_and_never_uses_search(cfg, tmp_path, monkeypatch):
    import gevd.evaluation.reporting as reporting
    monkeypatch.setattr(reporting, 'make_report', lambda *args, **kwargs: None)
    from gevd.training.runner import run_budgeted_main
    result = run_budgeted_main(cfg, 60)  # Explicit config must select GEVD automatically.
    assert result['total_calls'] == 60
    assert result['counts'].get('credit', 0) == 0
    assert result['counts'].get('mcbr', 0) == 0
    records = [json.loads(line) for line in (tmp_path / 'episodes.jsonl').read_text().splitlines()]
    assert records and all('retrospective_utility' in row for row in records)
    assert all(row['retrospective_utility']['max_qprime_to_q_ratio'] <= .1 + 1e-7 for row in records)
    restored = GEVDTrainer.from_config(cfg)
    restored.load_checkpoint(tmp_path / 'checkpoint.pt')
    assert restored.agent.auxiliary_updates > 0
    assert restored.agent.auxiliary_episodes == result['episodes']
    assert restored.agent.auxiliary_replay[-1].dones[-1] == (not records[-1]['budget_truncated'])


def test_reject_incompatible_search_and_trace_settings(cfg):
    cfg['event_aligned']['enabled'] = True
    with pytest.raises(ValueError, match='paired-search'):
        GEVDTrainer.from_config(cfg)
    cfg['event_aligned']['enabled'] = False
    cfg['retrospective_utility']['max_trace_steps'] = 2
    with pytest.raises(ValueError, match='horizon'):
        GEVDTrainer.from_config(cfg)
