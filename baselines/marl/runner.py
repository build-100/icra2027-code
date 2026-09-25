"""Independent v1 QMIX/COMA experiment; no GEVD learner imports."""
from pathlib import Path
import copy, hashlib, importlib.abc, json, math, os, random
import sys, threading, time, traceback
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generated'
FORBIDDEN=('gevd.training.base', 'gevd.training.gevd', 'gevd.training.search',
           'gevd.models.agent', 'gevd.models.credit')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONIOENCODING']='utf-8'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import numpy as np
import torch
import yaml

class RejectGEVDLearning(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in FORBIDDEN: raise ImportError('Forbidden GEVD learning module: '+fullname)
        return None

def setup():
    if not any(isinstance(finder, RejectGEVDLearning) for finder in sys.meta_path):
        sys.meta_path.insert(0,RejectGEVDLearning())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    from baselines.marl.networks import QMIX,COMA,IndependentLocalNetwork
    from baselines.marl.context import make_context
    from baselines.budget import SearchLedger
    from gevd.environments.multi_robot import GEVDMultiRobotEnv
    assert not set(FORBIDDEN).intersection(sys.modules)
    return QMIX,COMA,make_context,SearchLedger

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(20):
        try: os.replace(tmp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def epsilon(C):return 1.-.95*min(1.,max(0.,C/15000.))

def rates(learner):
    return {name:float(getattr(learner,name).param_groups[0]['lr'])
        for name in ('optimizer','actor_optimizer','critic_optimizer') if hasattr(learner,name)}

def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):visit(v.detach().cpu().numpy())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            for child in v:visit(child)
        else:h.update(repr(v).encode())
        h.update(b'\0')
    visit(value);return h.hexdigest()

def signature(learner):
    names=('online','target','mixer','target_mixer','critic','optimizer','actor_optimizer','critic_optimizer')
    result={name:digest(getattr(learner,name).state_dict()) for name in names if hasattr(learner,name)}
    result.update(updates=learner.updates,actor_updates=getattr(learner,'actor_updates',0),
        pending=digest(getattr(learner,'pending',[])))
    if hasattr(learner,'replay'):
        result['replay']=(len(learner.replay),learner.replay.position,repr(learner.replay.rng.getstate()))
    return result

def rollout(env,learner,ledger,exam=False):
    env.reset();initial=env.potential();routes=[[x] for x in env.start_labels]
    records=[];G=0.;distance=0.;calls=[];epsilons=[];raw_rewards=[]
    while not env.done and (exam or ledger.remaining>0):
        C=ledger.total;eps=0. if exam else epsilon(C)
        obs=learner.observations(env);mask=env.action_masks()
        actions=learner.actions(obs,mask,eps,exam=exam)
        assert not mask[np.arange(env.num_robots),actions].any()
        value=env.step(actions)
        reward=learner.training_reward(env,value)
        assert math.isclose(reward,value.reward,abs_tol=1e-12)
        if not exam:
            records.append(dict(obs=obs,mask=mask,actions=actions,reward=reward,
                next_obs=learner.observations(env),
                next_mask=np.ones_like(value.action_masks) if value.done else value.action_masks,
                done=value.done,epsilon=eps))
        calls.append(C);epsilons.append(eps);raw_rewards.append(float(reward))
        G+=reward;distance+=value.travel_distance
        for route,label in zip(routes,env.current_labels):route.append(label)
    S=env.structural_score();K=env.coverage_count();c=env.component_count();T=env.state.time
    assert math.isclose(G,env.potential()-initial-env.beta*distance,abs_tol=1e-9)
    row=dict(G=G,J=S-env.beta*distance,S=S,K=K,coverage=K,V=env.num_nodes,c=c,T=T,D=distance,
        success=bool(env.success),terminal=bool(env.done),budget_truncated=not env.done,
        routes=routes,initial_potential=initial,return_unscaled=G,
        failure_type=None if env.success else ('budget_truncated' if not env.done else
            'full_unfused' if K==env.num_nodes else 'missed_coverage' if c==1 else 'both'))
    if not exam:row.update(action_C=calls,action_epsilon=epsilons,raw_rewards=raw_rewards,
        optimizer_reward_scale=.1,scaled_reward_sum=.1*G)
    return row,records

def evaluate(context,learner,ledger,episode,marks):
    env=context.env;snapshot=env.snapshot();before=signature(learner)
    ledger_before=(ledger.total,len(ledger.events),len(ledger.rollouts),digest(ledger.best),len(ledger.unique))
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    local=copy.deepcopy(learner.rng.bit_generator.state) if isinstance(learner.rng,np.random.Generator) else learner.rng.getstate()
    mode_states=[(m,m.training) for name in ('online','target','mixer','target_mixer','critic')
        if hasattr(learner,name) for m in getattr(learner,name).modules()]
    try:
        with ledger.mode('evaluation'):row,_=rollout(env,learner,ledger,exam=True)
    finally:
        env.restore(snapshot);random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(rng[3])
        if isinstance(learner.rng,np.random.Generator):learner.rng.bit_generator.state=local
        else:learner.rng.setstate(local)
        for module,training in mode_states:module.training=training
    assert signature(learner)==before
    assert ledger_before==(ledger.total,len(ledger.events),len(ledger.rollouts),digest(ledger.best),len(ledger.unique))
    row.update(C=ledger.total,requested_marks=marks,episode=episode,epsilon=0.,
        optimizer_updates=learner.updates,actor_updates=getattr(learner,'actor_updates',None),
        learning_rates=rates(learner),training_state_preserved=True,candidate_eligible=False)
    return row

def monitor_start(folder):
    from baselines.marl import runtime_monitor as runtime
    runtime.dump=dump
    runtime.reserve_cpu_headroom()
    monitor=runtime.ResourceMonitor(folder/'runtime')
    while True:
        row=monitor.sample()
        if row['ram_available_gib']>=20 and row['gpu_temperature_c']<80 and row['gpu_memory_used_mib']<=.8*row['gpu_memory_total_mib']:break
        append(folder/'runtime/startup_wait.jsonl',row);time.sleep(5)
    monitor.start();return monitor

def run_one(job,preflight=False):
    QMIX,COMA,make_context,SearchLedger=setup()
    cfg=yaml.safe_load(Path(job['config']).read_text());folder=Path(job['directory'])
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'started.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),fresh=True),f)
    budget=cfg['search']['total_simulator_budget'];ledger=SearchLedger(budget,folder)
    monitor=None;began=time.time();episode=0;last=None
    try:
        if os.name=='nt' and cfg['vdn']['device'].startswith('cuda'):monitor=monitor_start(folder)
        elif os.name=='nt':
            from baselines.marl.runtime_monitor import reserve_cpu_headroom
            reserve_cpu_headroom()
        context=make_context(cfg);learner=(QMIX if job['method']=='qmix' else COMA)(context)
        assert type(learner.online).__name__=='IndependentLocalNetwork'
        assert not hasattr(context,'agent') and not set(FORBIDDEN).intersection(sys.modules)
        expected={'optimizer':5e-5} if job['method']=='qmix' else {'actor_optimizer':5e-4,'critic_optimizer':5e-4}
        assert rates(learner)==expected
        assert isinstance(learner.optimizer,torch.optim.Adam) if job['method']=='qmix' else (
            isinstance(learner.actor_optimizer,torch.optim.RMSprop) and isinstance(learner.critic_optimizer,torch.optim.RMSprop))
        dump(folder/'initialization.json',dict(**learner.input_metadata(),method=job['method'],seed=job['seed'],
            learner_class=type(learner).__name__,network_class=type(learner.online).__name__,
            central_class=type(learner.mixer if job['method']=='qmix' else learner.critic).__name__,
            optimizers={name:type(getattr(learner,name)).__name__ for name in expected},learning_rates=expected,
            forbidden_modules_loaded=[],fresh=True,loaded_old_model=False,optimizer_reward_scale=.1))
        next_exam=500;next_checkpoint=5000;train=[];exams=[]
        with ledger.installed():
            last=evaluate(context,learner,ledger,episode,[0]);exams.append(last);append(folder/'exams.jsonl',last)
            while ledger.remaining:
                if monitor:monitor.gate()
                episode+=1;ledger.episode=episode;begin_C=ledger.total
                trajectory=ledger.trajectory('factual',[[x] for x in context.env.start_labels])
                with ledger.mode('factual',trajectory):row,records=rollout(context.env,learner,ledger)
                # Constant schedules evaluated against actual C at the update boundary.
                for name,lr in expected.items():
                    for group in getattr(learner,name).param_groups:group['lr']=lr
                losses=learner.update(records,force=not ledger.remaining) if job['method']=='coma' else learner.update(records)
                for diagnostic in losses:
                    assert all(math.isfinite(v) for v in diagnostic.values())
                    append(folder/'losses.jsonl',dict(C=ledger.total,episode=episode,**diagnostic))
                row.update(C=ledger.total,C_begin=begin_C,episode=episode,factual_steps=ledger.counts['factual'],
                    learning_rates=rates(learner),epsilon_begin=epsilon(begin_C),epsilon_next=epsilon(ledger.total),
                    optimizer_updates=learner.updates,actor_updates=getattr(learner,'actor_updates',None),
                    update_losses=losses,mean_loss={key:float(np.mean([r[key] for r in losses])) for key in losses[0]} if losses else None)
                append(folder/'episodes.jsonl',row);train.append(row)
                if ledger.total>=next_exam or not ledger.remaining:
                    marks=[]
                    while next_exam<=ledger.total:marks.append(next_exam);next_exam+=500
                    if not ledger.remaining and budget not in marks:marks.append(budget)
                    last=evaluate(context,learner,ledger,episode,marks);exams.append(last);append(folder/'exams.jsonl',last)
                    print(json.dumps(dict(method=job['method'],map=job['map'],N=job['N'],seed=job['seed'],
                        C=ledger.total,G=last['G'],K=last['K'],c=last['c'],success=last['success'])),flush=True)
                dump(folder/'progress.json',dict(status='running',pid=os.getpid(),C=ledger.total,budget=budget,
                    episode=episode,optimizer_updates=learner.updates,actor_updates=getattr(learner,'actor_updates',None),
                    last_loss=row['mean_loss'],learning_rates=rates(learner),epsilon=epsilon(ledger.total),
                    current_policy=last,counts=dict(ledger.counts),seconds=time.time()-began,updated=time.time()))
                if ledger.total>=next_checkpoint or not ledger.remaining:
                    cp=folder/'checkpoints';cp.mkdir(exist_ok=True)
                    torch.save(dict(learner=learner.checkpoint(),config=cfg,C=ledger.total),cp/f'C_{ledger.total:05d}.pt')
                    while next_checkpoint<=ledger.total:next_checkpoint+=5000
            assert ledger.total==budget
            torch.save(dict(learner=learner.checkpoint(),config=cfg,C=ledger.total),folder/'final_model.pt')
            final_cp_digest=digest(learner.checkpoint()) if preflight else None
            last_audit=evaluate(context,learner,ledger,episode,[])
            assert last_audit['routes']==last['routes'] and math.isclose(last_audit['G'],last['G'],abs_tol=1e-9)
            if preflight:assert digest(learner.checkpoint())==final_cp_digest
        # Audit is recorded separately; the extra reproducibility rollout is not an evaluation checkpoint.
        ledger.counts['evaluation']-=last_audit['T'];ledger.counts['audit']+=last_audit['T']
        best=copy.deepcopy(ledger.best)
        if best:best.update(G=best['J']+cfg['reward']['rho_v']*(context.env.num_nodes-job['N'])+cfg['reward']['rho_g']*(job['N']-1),K=best['coverage'])
        dump(folder/'best_verified_feasible.json',best)
        result=dict(method=job['method'],map=job['map'],N=job['N'],seed=job['seed'],C=ledger.total,
            current_policy=last,best_verified_feasible=best,counts=dict(ledger.counts),episodes=episode,
            optimizer_updates=learner.updates,actor_updates=getattr(learner,'actor_updates',None),
            learning_rates=rates(learner),optimizer_reward_scale=.1,seconds=time.time()-began,
            final_model=str(folder/'final_model.pt'),old_model_loaded=False,
            interpretation='Stable performance was not observed within this budget' if not all(r['success'] for r in exams if r['C']>=.8*budget) else
                'All recorded evaluations in the last 20% of the budget succeeded; this does not establish convergence')
        dump(folder/'result.json',result)
        audit_job(job)
        dump(folder/'complete.json',dict(passed=True,C=ledger.total,finished=time.time()))
        state=read(folder/'progress.json');state.update(status='complete');dump(folder/'progress.json',state)
    except BaseException:
        dump(folder/'error.json',dict(error=traceback.format_exc(),C=ledger.total));raise
    finally:
        if monitor:monitor.close()

def audit_job(job):
    folder=Path(job['directory']);cfg=yaml.safe_load(Path(job['config']).read_text());budget=cfg['search']['total_simulator_budget']
    train=rows(folder/'episodes.jsonl');exams=rows(folder/'exams.jsonl');result=read(folder/'result.json')
    assert result['C']==budget and sum(result['counts'].get(k,0) for k in
        ('factual','credit','mcbr','planner','candidate_validation'))==budget
    assert all(result['counts'].get(k,0)==0 for k in ('credit','mcbr','planner'))
    assert exams[0]['C']==0 and exams[-1]['C']==budget
    assert sorted({m for r in exams for m in r['requested_marks']})==sorted(set([0,budget]+list(range(500,budget+1,500))))
    for r in exams:
        assert r['epsilon']==0 and r['training_state_preserved'] and not r['candidate_eligible']
        assert all(mark<=r['C'] and r['C']-mark<2*cfg['environment']['t_max'] for mark in r['requested_marks'])
    for r in train+exams:
        expected=r['S']+cfg['reward']['rho_v']*(r['K']-job['N'])+cfg['reward']['rho_g']*(job['N']-r['c'])-cfg['reward']['beta']*r['D']
        assert math.isclose(expected,r['G'],abs_tol=1e-8)
        assert r['learning_rates']==({'optimizer':5e-5} if job['method']=='qmix' else {'actor_optimizer':5e-4,'critic_optimizer':5e-4})
    for r in train:
        assert all(math.isclose(e,epsilon(c),abs_tol=1e-12) for c,e in zip(r['action_C'],r['action_epsilon']))
        assert math.isclose(sum(r['raw_rewards']),r['G'],abs_tol=1e-9)
        assert math.isclose(r['scaled_reward_sum'],.1*r['G'],abs_tol=1e-9)
    from baselines.marl.networks import QMIX,COMA
    from baselines.marl.context import make_context
    from baselines.budget import SearchLedger
    # Load only this run's newly saved final model for independent final-policy verification.
    ctx=make_context(cfg);learner=(QMIX if job['method']=='qmix' else COMA)(ctx)
    from baselines.checkpoints import load_checkpoint
    cp=load_checkpoint(folder/'final_model.pt',map_location=ctx.device)
    learner.load_checkpoint(cp['learner']);ledger=SearchLedger(budget)
    with ledger.installed():final=evaluate(ctx,learner,ledger,0,[])
    assert final['routes']==exams[-1]['routes'] and math.isclose(final['G'],exams[-1]['G'],abs_tol=1e-9)
    best=result['best_verified_feasible'];best_calls=0
    if best:
        from baselines.budget import state_digest
        with ledger.installed(),ledger.mode('audit'):
            ctx.env.reset()
            for t in range(best['T']):
                ctx.env.step([ctx.env.prior.action_for_neighbor(r[t],r[t+1]) for r in best['routes']]);best_calls+=1
        assert ctx.env.success and state_digest(ctx.env.state)==best['state_sha256'] and best['C']<=budget
    late=[r for r in exams if r['C']>=.8*budget]
    dump(folder/'audit.json',dict(passed=True,seed=job['seed'],checkpoints=len(exams),
        final_model_reproduced=True,readonly_extra_calls=final['T']+best_calls,
        no_gevd_learning_modules=not set(FORBIDDEN).intersection(sys.modules),
        late_successes=sum(r['success'] for r in late),late_evaluations=len(late)))
