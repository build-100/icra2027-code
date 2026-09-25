import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
from gevd.models.dqn import DQNAgent
from gevd.training.gevd import GEVDTrainer
from ablations.variants import GEVDAblationTrainer, IndependentTDPrimary, LABELS
from gevd.training.base import _network_digest
from gevd.training.replay import DQNTransition
from ablations.raw_structure import raw_structural_score
from gevd.training.retrospective_utility import collect_auxiliary_episode


def config(tmp_path, variant):
    cfg = yaml.safe_load((ROOT/'configs/simulation/comparison/map8/N3/gevd.yaml').read_text())
    cfg['full_gevd_ablation'] = variant
    cfg['vdn'].update(device='cpu', hidden_dim=16, batch_size=2, min_factual_replay=2)
    cfg['retrospective_utility']['batch_size'] = 2
    cfg['output']['directory'] = str(tmp_path)
    cfg['environment']['graph_path'] = str(ROOT / cfg['environment']['graph_path'])
    return cfg


def test_independent_td_targets_are_local_masked_and_terminal_safe():
    net = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[1.], [2.]]))
    agent = IndependentTDPrimary(DQNAgent(1, 2, 2, .001, 1., 10, 10., 1,
        network=net, loss_kind='mse', td_reward_scale=.1, double_dqn=True))
    with torch.no_grad():
        agent.target.weight.copy_(torch.tensor([[10.], [20.]]))
    batch = DQNTransition(
        state=np.array([[[1.], [3.]], [[2.], [4.]]]), action=np.array([[0, 1], [1, 0]]),
        reward=np.array([5., 7.]), next_state=np.array([[[2.], [4.]], [[999.], [999.]]]),
        action_mask=np.zeros((2, 2, 2), bool),
        next_action_mask=np.array([[[False, True], [False, False]], [[True, True], [True, True]]]),
        done=np.array([False, True]), reward_channels=({}, {}))
    predicted = np.array([1., 6., 4., 4.])
    targets = np.array([20.5, 80.5, .7, .7])
    loss = agent.learn(batch)
    np.testing.assert_allclose(agent.last_predicted_q, predicted)
    np.testing.assert_allclose(agent.last_td_targets, targets)
    assert loss == pytest.approx(np.mean((targets - predicted)**2))
    assert agent.update_steps == 1
    assert 'independent_td_loss' in agent.diagnostic_history[-1]


@pytest.mark.parametrize('variant', LABELS)
def test_identical_initialization_encoders_and_auxiliary(tmp_path, variant):
    cfg = config(tmp_path, variant)
    tr = GEVDAblationTrainer.from_config(cfg)
    base_cfg = copy.deepcopy(cfg); base_cfg.pop('full_gevd_ablation')
    base = GEVDTrainer.from_config(base_cfg)
    assert tr.env.schema() == base.env.schema()
    for name in ('online', 'target', 'auxiliary', 'auxiliary_target'):
        assert _network_digest(getattr(tr.agent, name)) == _network_digest(getattr(base.agent, name))
    assert tr.agent.settings == base.agent.settings


@pytest.mark.parametrize('variant', LABELS)
def test_reward_replacement_only_in_primary_replay(tmp_path, variant):
    from types import SimpleNamespace
    from tests.synthetic_routes import three_robot_success
    cfg, routes = three_robot_success(config(tmp_path, variant))
    tr = GEVDAblationTrainer.from_config(cfg)
    tr._previous_raw_score = tr._ablation_training_return = 0.
    common = 0.; cache = []; transitions = []
    for t, labels in enumerate(zip(*[route[1:] for route in routes])):
        obs = tr.env.local_observations(); masks = tr.env.action_masks()
        actions = [tr.env.prior.action_for_neighbor(a,b) for a,b in zip(tr.env.current_labels, labels)]
        cache.append(SimpleNamespace(state=tr.env.snapshot(), absolute_time=t,
            observations=obs, action_masks=masks, factual_action=actions))
        result = tr.env.step(actions); transitions.append(result); common += result.reward
        previous_raw = tr._previous_raw_score
        tr._push(tr.factual_replay, obs, actions, result, masks)
        saved = tr.factual_replay.memory[-1]
        channels = result.reward_channels.as_dict()
        if variant == 'raw_structure':
            channels['spectral'] = raw_structural_score(tr.env) - previous_raw
        if variant == 'no_gauge':
            channels['gauge'] = 0.
        assert saved.reward == pytest.approx(sum(channels.values()))
        assert saved.reward_channels == pytest.approx(channels)
    assert tr.env.success
    aux = collect_auxiliary_episode(cache, transitions, [0.] * len(transitions))
    assert aux.rewards.sum() == pytest.approx(.5)
    expected = common
    if variant == 'no_gauge': expected -= 2.
    if variant == 'raw_structure':
        diag = tr.env.information_diagnostics()
        expected += diag['auxiliary_conditional_score']
    assert tr._ablation_training_return == pytest.approx(expected)


@pytest.mark.parametrize('variant', LABELS)
def test_pipeline_checkpoint_auxiliary_and_cross_variant_rejection(tmp_path, monkeypatch, variant):
    import gevd.evaluation.reporting as reporting
    from gevd.training.runner import run_budgeted_main
    monkeypatch.setattr(reporting, 'make_report', lambda *a, **kw: None)
    cfg = config(tmp_path, variant)
    result = run_budgeted_main(cfg, 60)
    assert result['total_calls'] == 60
    assert not result['counts'].get('credit', 0)
    episodes = [json.loads(x) for x in (tmp_path/'episodes.jsonl').read_text().splitlines()]
    assert all('retrospective_utility' in x and 'ablation_training' in x for x in episodes)
    tr = GEVDAblationTrainer.from_config(cfg); tr.load_checkpoint(tmp_path/'checkpoint.pt')
    assert tr.agent.auxiliary_updates == tr.agent.update_steps > 0
    assert tr.agent.auxiliary_episodes == len(episodes)
    assert tr.run_factual_episode(training=False)['routes'] == result['current_policy']['routes']
    other = copy.deepcopy(cfg); other['full_gevd_ablation'] = next(x for x in LABELS if x != variant)
    with pytest.raises(ValueError, match='ablation'):
        GEVDAblationTrainer.from_config(other).load_checkpoint(tmp_path/'checkpoint.pt')
    other.pop('full_gevd_ablation')
    with pytest.raises(ValueError, match='ablation'):
        GEVDTrainer.from_config(other).load_checkpoint(tmp_path/'checkpoint.pt')
