"""Budgeted GEVD training and learning-curve logging."""
import copy
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import yaml


def run_budgeted_main(config, budget, *, trainer_class=None):
    from gevd.training.search import BudgetedSearchTrainer, SearchLedger
    from gevd.evaluation.replay import preserved_readonly, simple_episode
    from gevd.evaluation.reporting import make_report
    from gevd.training.runtime import ResourceMonitor, reserve_cpu_headroom, dump, append
    if isinstance(budget, bool) or int(budget) != budget or budget <= 0:
        raise ValueError('Search budget must be a positive integer.')
    cfg=copy.deepcopy(config)
    cfg.setdefault('search',{})['total_simulator_budget']=int(budget)
    cfg.setdefault('event_aligned',{}).setdefault('paired_return',{})['cache_factual_returns']=True
    torch.set_num_threads(int(cfg['vdn'].get('cpu_threads',1)))
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if trainer_class is None and cfg.get('full_gevd_ablation'):
        from ablations.variants import GEVDAblationTrainer
        trainer_class = GEVDAblationTrainer
    if trainer_class is None and cfg.get('retrospective_utility', {}).get('enabled', False):
        from gevd.training.gevd import GEVDTrainer
        trainer_class = GEVDTrainer
    if trainer_class is None and cfg.get('training_structure') == 'componentwise_raw':
        from ablations.raw_structure import RawStructureTrainer
        trainer_class = RawStructureTrainer
    trainer_type = BudgetedSearchTrainer if trainer_class is None else trainer_class
    tr=trainer_type.from_config(cfg)
    out=Path(tr.output_directory);out.mkdir(parents=True,exist_ok=True)
    # Never reuse an existing run, even when it stopped before the final checkpoint.
    with (out/'search_started.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),time=time.time(),budget=budget,seed=tr.seed,
                       fresh=True,variant=cfg.get('experiment', {}).get('variant', 'GEVD'),stop_unit='search_simulator_calls'),stream)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    ledger=SearchLedger(budget,out);tr.bind_search(ledger)
    schedule = cfg.get('epsilon_schedule')
    if schedule:
        tr.epsilon = lambda: float(schedule['start'] + min(1., ledger.total / schedule['decay_end_calls']) * (schedule['end'] - schedule['start']))
    def update_learning_rate():
        rule = cfg.get('learning_rate_schedule')
        if rule:
            fraction = min(1., max(0., (ledger.total - rule['decay_start_calls']) / (rule['decay_end_calls'] - rule['decay_start_calls'])))
            value = rule['initial'] + fraction * (rule['final'] - rule['initial'])
            for group in tr.agent.optimizer.param_groups:
                group['lr'] = value
        return float(tr.agent.optimizer.param_groups[0]['lr'])
    began=time.perf_counter();monitor=None
    state=dict(status='initializing',pid=os.getpid(),C=0,budget=budget,episode=0)
    dump(out/'progress.json',state)
    try:
        if tr.device.type=='cuda' and os.name=='nt':
            reserve_cpu_headroom();monitor=ResourceMonitor(out/'runtime');monitor.start()
        (out/'effective_config.yaml').write_text(yaml.safe_dump(tr.config,sort_keys=False),encoding='utf-8')
        exams=[]
        def evaluate():
            with preserved_readonly(tr,ledger):
                with ledger.mode('evaluation'):
                    result=tr.run_factual_episode(training=False)
                row=simple_episode(result,tr.episode_count,tr.global_step,tr.env.beta,tr.env.num_nodes)
                if 'raw_structure' in result:
                    row['raw_structure'] = result['raw_structure']
                if 'ablation_training' in result:
                    row['ablation_training'] = result['ablation_training']
                row.update(C=ledger.total,optimizer_updates=tr.agent.update_steps,epsilon=0.,
                           training_state_preserved=True,candidate_eligible=False)
            append(out/'exams.jsonl',row);exams.append(row);return row
        next_report=1000;next_checkpoint=5000
        with ledger.installed():
            evaluate()
            while ledger.remaining:
                episode_index=tr.episode_count;epsilon_used=tr.epsilon()
                if monitor is not None:monitor.gate()
                ledger.episode=episode_index+1
                with ledger.mode('factual',ledger.trajectory('factual',[[s] for s in tr.env.start_labels])):
                    value=tr.run_factual_episode(max_steps=ledger.remaining)
                if value['terminal']:tr.generate_branches(value['pre_action_cache'])
                learning_rate=update_learning_rate()
                losses=tr._updates_if_ready(value['joint_steps'])
                row=simple_episode(value,tr.episode_count,tr.global_step,tr.env.beta,tr.env.num_nodes)
                row.update(C=ledger.total,learning_rate=learning_rate,epsilon=epsilon_used,epsilon_next=tr.epsilon(),episode_index=episode_index,
                           optimizer_updates=tr.agent.update_steps,episode_optimizer_updates=len(losses),
                           mean_training_loss=float(np.mean(losses)) if losses else None)
                if 'retrospective_utility' in value:
                    row['retrospective_utility'] = value['retrospective_utility']
                if 'raw_structure' in value:
                    row['raw_structure'] = value['raw_structure']
                if 'ablation_training' in value:
                    row['ablation_training'] = value['ablation_training']
                append(out/'episodes.jsonl',row)
                for diagnostic in tr.agent.diagnostic_history:
                    diagnostic.update(episode=tr.episode_count,C=ledger.total)
                    append(out/'optimizer_diagnostics.jsonl',diagnostic)
                tr.agent.diagnostic_history.clear()
                event_audit = getattr(tr.event_credit, 'audit', None)
                if event_audit is not None:
                    for label in event_audit:append(out/'paired_labels.jsonl',label)
                    event_audit.clear()
                last=evaluate()
                ledger.emit('search_progress.jsonl',dict(C=ledger.total,episode=tr.episode_count,
                            best_J=ledger.best['J'] if ledger.best else None))
                state.update(status='running',C=ledger.total,episode=tr.episode_count,epsilon=epsilon_used,
                    factual_steps=tr.global_step,optimizer_updates=tr.agent.update_steps,
                    target_sync_count=len(tr.agent.target_sync_steps),last_training_loss=row['mean_training_loss'],
                    current_policy=last,counts=dict(ledger.counts),elapsed_seconds=time.perf_counter()-began,
                    updated_at=time.time())
                dump(out/'progress.json',state)
                if ledger.total>=next_report or not ledger.remaining:
                    make_report(out,exams)
                    print(json.dumps(dict(C=ledger.total,budget=budget,episode=tr.episode_count,
                          coverage=last['coverage'],success=last['success'],G=last['return'])),flush=True)
                    while next_report<=ledger.total:next_report+=1000
                if ledger.total>=next_checkpoint or not ledger.remaining:
                    tr.save_checkpoint(out/'checkpoints'/f'C_{ledger.total:05d}.pt')
                    dump(out/'search_state.json',dict(episode=tr.episode_count,counts=dict(ledger.counts),best=ledger.best))
                    while next_checkpoint<=ledger.total:next_checkpoint+=5000
            assert ledger.total==budget and len(exams)==tr.episode_count+1
            tr.save_checkpoint(out/'checkpoint.pt')
        result=dict(success=ledger.best is not None,selected=ledger.best,counts=dict(ledger.counts),
            total_calls=ledger.total,budget=budget,post_training_optimization=False,
            current_policy=exams[-1],episodes=tr.episode_count,optimizer_updates=tr.agent.update_steps,
            checkpoint=str(out/'checkpoint.pt'),learning_curve=str(out/'report/learning_curve_w49.png'))
        dump(out/'search_result.json',result);dump(out/'best_candidate.json',ledger.best)
        # Retain the existing CLI result interface as well as the complete resumable checkpoint.
        torch.save(dict(agent=tr.agent.state_dict(),config=tr.config,total_calls=ledger.total),out/'search_checkpoint.pt')
        state.update(status='complete',completed_at=time.time());dump(out/'complete.json',state);dump(out/'progress.json',state)
        return result
    except BaseException:
        state.update(status='failed',error=traceback.format_exc())
        dump(out/'failed.json',state);dump(out/'progress.json',state)
        tr.save_checkpoint(out/'failure_checkpoint.pt')
        raise
    finally:
        if monitor is not None:monitor.stop_event.set()
