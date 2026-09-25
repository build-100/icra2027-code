import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[1]
from ablations.raw_structure import RawStructureTrainer,raw_structural_score
from gevd.training.search import BudgetedSearchTrainer, SearchLedger
from gevd.evaluation.replay import replay_metrics

def config(tmp_path):
    cfg=yaml.safe_load((ROOT/'configs/simulation/ablation/map8_N3/no_retrospective.yaml').read_text(encoding='utf-8'))
    cfg['training_structure']='componentwise_raw'
    cfg['vdn'].update(device='cpu',hidden_dim=16,batch_size=2,min_factual_replay=2)
    cfg['output']['directory']=str(tmp_path)
    cfg['environment']['graph_path']=str(ROOT/cfg['environment']['graph_path'])
    return cfg

def test_eq5_and_only_replay_spectral_replaced_on_known_success(tmp_path):
    from tests.synthetic_routes import three_robot_success
    cfg,routes=three_robot_success(config(tmp_path))
    cfg['environment']['graph_path']=str(ROOT/cfg['environment']['graph_path'])
    tr=RawStructureTrainer.from_config(cfg)
    labels=iter(zip(*[r[1:] for r in routes]))
    def actions(*args,**kwargs):
        return np.array([tr.env.prior.action_for_neighbor(a,b) for a,b in zip(tr.env.current_labels,next(labels))])
    tr.agent.select_actions=actions
    ledger=SearchLedger(100)
    tr.bind_search(ledger)
    with ledger.installed(), ledger.mode('factual',ledger.trajectory('factual',[[x] for x in tr.env.start_labels])):
        episode=tr.run_factual_episode()
    assert episode['success']
    assert ledger.best['S']==episode['final_structural_score']
    assert ledger.counts['candidate_validation']==episode['joint_steps']
    native=episode['raw_structure'];diag=tr.env.information_diagnostics()
    assert native['S_raw']==pytest.approx(diag['raw_score_same_factor_graph'])
    assert native['S_raw']==pytest.approx(episode['final_structural_score']+diag['auxiliary_conditional_score'])
    assert abs(native['S_raw']-episode['final_structural_score'])>.001
    assert sum(x.reward for x in tr.factual_replay.memory)==pytest.approx(native['training_return'])
    previous=0.
    for saved,result in zip(tr.factual_replay.memory,episode['transitions']):
        raw=raw_structural_score(tr.env,result.state)
        assert saved.reward_channels['spectral']==pytest.approx(raw-previous)
        for key in ('coverage','gauge','travel'):
            assert saved.reward_channels[key]==result.reward_channels.as_dict()[key]
        previous=raw
    common=replay_metrics(cfg,routes,True)
    assert common['S']==episode['final_structural_score']

def test_common_scores_and_checkpoint_roundtrip(tmp_path,monkeypatch):
    import gevd.evaluation.reporting as reporting
    monkeypatch.setattr(reporting,'make_report',lambda *a,**kw:None)
    from gevd.training.runner import run_budgeted_main
    cfg=config(tmp_path);result=run_budgeted_main(cfg,60)
    assert result['total_calls']==60
    saved=json.loads((tmp_path/'exams.jsonl').read_text().splitlines()[-1])
    assert 'raw_structure' in saved and saved['raw_structure']['common_S']==saved['S']
    tr=RawStructureTrainer.from_config(cfg);tr.load_checkpoint(tmp_path/'checkpoint.pt')
    episode=tr.run_factual_episode(training=False)
    assert episode['routes']==saved['routes']
    assert episode['return']==pytest.approx(saved['G'])
    primary=copy.deepcopy(cfg);primary.pop('training_structure')
    base=BudgetedSearchTrainer.from_config(primary)
    with pytest.raises(ValueError,match='Training structure'):
        base.load_checkpoint(tmp_path/'checkpoint.pt')

def test_main_configuration_promoted_to_full_gevd(tmp_path):
    from gevd.training.gevd import GEVDTrainer
    cfg=yaml.safe_load((ROOT/'configs/simulation/comparison/map8/N3/gevd.yaml').read_text(encoding='utf-8'))
    cfg['vdn'].update(device='cpu',hidden_dim=16)
    cfg['output']['directory']=str(tmp_path)
    tr=GEVDTrainer.from_config(cfg)
    assert tr.algorithm_name=='GEVD'
    assert tr.agent.settings['execution_lambda']==.1
