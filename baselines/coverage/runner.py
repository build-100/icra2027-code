"""Seven-node QMIX coverage objectives, schedules and policy evaluation."""
from pathlib import Path
import json
import math
import os
import random
import time
import traceback
from baselines.budget import SearchLedger
from baselines.coverage.networks import QMIX, network_digest
from baselines.coverage.reward import coverage_potential, intra_information
from gevd.environments.prior_graph import PriorGraph
from gevd.environments.multi_robot import GEVDMultiRobotEnv

def imports(_):
    return (None, SearchLedger, QMIX, network_digest)

def train_potential(row, method):
    return coverage_potential(row, method, intra_weight=7.0)

def create(out, method, seed, device=None):
    import yaml
    import numpy as np
    import torch
    from types import SimpleNamespace
    cfg = yaml.safe_load((out / 'configs' / f'{method}_{seed}.yaml').read_text(encoding='utf-8'))
    if method not in ('coverage_only', 'coverage_first'):
        raise ValueError('Coverage method must be coverage_only or coverage_first')
    if device:
        cfg['vdn']['device'] = device
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg['vdn']['device'])
    e, r = (cfg['environment'], cfg['reward'])
    prior = PriorGraph(e['graph_path'], e['map_width'], e['start_nodes'][0], device)
    env = GEVDMultiRobotEnv(prior, tuple(e['start_nodes']), e['t_max'], r['alpha'], r['beta'], r['rho_v'], r['rho_g'], r['pair_factor_weight'])
    learner = QMIX(SimpleNamespace(config=cfg, env=env, seed=seed, device=device))
    learner.environment_schema['experiment_objective'] = method
    return (None, env, learner, cfg)

def exam(env, learner, tr, method, calls, digest):
    import numpy as np
    import torch
    state = env.snapshot()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), learner.rng.getstate())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modules = [learner.online, learner.target, learner.mixer, learner.target_mixer]
    hashes = [digest(module) for module in modules]
    counters = (learner.updates, learner.completed_episodes, learner.update_credit, len(learner.replay))
    try:
        row = rollout(env, learner, method, calls, 0, exam=True)[0]
    finally:
        env.restore(state)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        learner.rng.setstate(rng[3])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        assert hashes == [digest(module) for module in modules]
        assert counters == (learner.updates, learner.completed_episodes, learner.update_credit, len(learner.replay))
    row.update(source='current_greedy_policy', candidate_eligible=False, training_state_preserved=True)
    return row

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(data, encoding='utf-8')
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.025 * (attempt + 1))
    path.write_text(data, encoding='utf-8')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def append(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def rows(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    result = []
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return result

def lr_at(calls, budget):
    fraction = min(1.0, max(0.0, (calls / budget - 0.5) / 0.3))
    return 5e-06 + fraction * (5e-07 - 5e-06)

def epsilon_at(calls, budget):
    fraction = min(1.0, max(0.0, calls / (0.6 * budget)))
    return 1.0 + fraction * (0.05 - 1.0)

def common_potential(row):
    return row['S'] - row['beta'] * row['D'] + row['coverage'] - 1.0 * (row['c'] - 1)

def should_select(row, best):
    return bool(row['success'] and (best is None or row['training_return'] > best['training_return'] + 1e-12))

def metrics(env):
    import numpy as np
    return dict(S=float(env.structural_score()), D=float(np.sum(env.state.M * env.edge_distances[None, :])), coverage=int(env.coverage_count()), c=int(env.component_count()), V=env.num_nodes, T=int(env.state.time), success=bool(env.success), beta=env.beta, distance_bound=float(env.num_robots * env.t_max * max(env.edge_distances)), **intra_information(env))

def finish_row(initial, final, method, routes, steps, total, terminal):
    expected = train_potential(final, method) - train_potential(initial, method)
    common = common_potential(final) - common_potential(initial)
    assert math.isclose(total, expected, abs_tol=1e-09), (total, expected)
    assert math.isclose(sum((s['common_reward'] for s in steps)), common, abs_tol=1e-09)
    return dict(**final, method=method, routes=routes, step_metrics=steps, training_return=float(total), common_G=common, initial=initial, terminal=terminal, budget_truncated=not terminal, **{'return': float(total)})

def rollout(env, learner, method, calls, remaining, exam=False):
    import numpy as np
    env.reset()
    before = initial = metrics(env)
    records, step_rows = ([], [])
    routes = [[s] for s in env.start_labels]
    total = 0.0
    for _ in range(env.t_max if exam else min(env.t_max, remaining)):
        obs, mask = (learner.observations(env), env.action_masks())
        epsilon = learner.epsilon(calls + len(records), exam=exam)
        actions = learner.actions(obs, mask, epsilon, exam=exam)
        transition = env.step(actions)
        after = metrics(env)
        common = common_potential(after) - common_potential(before)
        reward = train_potential(after, method) - train_potential(before, method)
        records.append(dict(obs=obs, mask=mask, actions=actions, reward=reward, next_obs=learner.observations(env), next_mask=np.ones_like(mask) if transition.done else transition.action_masks, done=transition.done, epsilon=epsilon))
        step_rows.append(dict(**after, training_reward=reward, common_reward=common))
        for i, label in enumerate(env.current_labels):
            routes[i].append(label)
        before = after
        total += reward
        if transition.done:
            break
    return (finish_row(initial, metrics(env), method, routes, step_rows, total, bool(env.done)), records)

def worker(out, method, seed, budget=20000, device=None):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    import torch
    torch.set_num_threads(1)
    Trainer, Ledger, QMIX, digest = imports(out)
    directory = out / method / f'seed_{seed}' / 'run'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'started.json').open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), time=time.time(), budget=budget), f)
    tr, env, learner, cfg = create(out, method, seed, device)
    ledger = Ledger(budget, directory)
    learner.epsilon = lambda calls, exam=False: 0.0 if exam else epsilon_at(calls, budget)
    save(directory / 'initialization.json', dict(method=method, seed=seed, initial_online=digest(learner.online), initial_mixer=digest(learner.mixer), device=str(learner.device), torch_version=torch.__version__, config=cfg))
    episodes = factual = 0
    best = None
    began = time.perf_counter()

    def do_exam(lr):
        with ledger.mode('evaluation'):
            row = exam(env, learner, tr, method, ledger.total, digest)
        row.update(C=ledger.total, factual_steps=factual, episode=episodes, seed=seed, epsilon=0.0, learning_rate=lr, optimizer_updates=learner.updates)
        append(directory / 'exams.jsonl', row)
        return row

    def policy_checkpoint(row):
        assert row['success']
        name = f"episode_{episodes:06d}_C_{ledger.total:06d}_G_{row['training_return']:.8f}.pt"
        folder = directory / 'late_best_checkpoints'
        folder.mkdir(exist_ok=True)
        payload = dict(method=method, seed=seed, config=cfg, evaluation=row, online=learner.online.state_dict(), checkpoint_kind='selected_policy_inference_only')
        payload['mixer'] = learner.mixer.state_dict()
        torch.save(payload, folder / name)
        return dict(episode=episodes, C=ledger.total, training_return=row['training_return'], common_G=row['common_G'], success=row['success'], coverage=row['coverage'], c=row['c'], path=str(folder / name), online_sha256=digest(learner.online))
    try:
        with ledger.installed():
            current = do_exam(lr_at(0, budget))
            while ledger.remaining:
                episodes += 1
                ledger.episode = episodes
                episode_start_C = ledger.total
                with ledger.mode('factual'):
                    row, records = rollout(env, learner, method, ledger.total, ledger.remaining)
                factual += len(records)
                epsilon = records[0]['epsilon']
                lr = lr_at(ledger.total, budget)
                for group in learner.optimizer.param_groups:
                    group['lr'] = lr
                diagnostics = learner.update(records)
                losses = [d['loss'] for d in diagnostics]
                updates = learner.updates
                row.update(C=ledger.total, factual_steps=factual, episode=episodes, seed=seed, episode_start_C=episode_start_C, epsilon=epsilon, learning_rate=lr, optimizer_updates=updates, mean_loss=sum(losses) / len(losses) if losses else None)
                append(directory / 'episodes.jsonl', row)
                current = do_exam(lr)
                if ledger.total >= 0.8 * budget:
                    if should_select(current, best):
                        best = policy_checkpoint(current)
                        append(directory / 'selection_events.jsonl', best)
                    append(directory / 'retained_curve.jsonl', dict(C=ledger.total, episode=episodes, available=best is not None, training_return=best['training_return'] if best else None, common_G=best['common_G'] if best else None, selected_episode=best['episode'] if best else None, selected_C=best['C'] if best else None, success=True if best else None, coverage=best['coverage'] if best else None, c=best['c'] if best else None))
                if episodes % 25 == 0 or not ledger.remaining:
                    save(directory / 'progress.json', dict(status='running', method=method, seed=seed, C=ledger.total, budget=budget, episode=episodes, factual_steps=factual, optimizer_updates=updates, lr=lr, epsilon=epsilon, counts=dict(ledger.counts), current=current, selected=best, elapsed_seconds=time.perf_counter() - began, updated_at=time.time()))
            if current['success']:
                torch.save(dict(learner=learner.checkpoint(), config=cfg, episode=episodes, counts=dict(ledger.counts)), directory / 'final_checkpoint.pt')
        result = dict(method=method, seed=seed, C=ledger.total, budget=budget, episodes=episodes, counts=dict(ledger.counts), optimizer_updates=updates, final=current, selected=best, elapsed_seconds=time.perf_counter() - began, search_best=None)
        assert ledger.total == budget
        save(directory / 'result.json', result)
        save(directory / 'complete.json', dict(status='complete', time=time.time(), C=ledger.total))
        print(json.dumps(dict(method=method, seed=seed, C=ledger.total, final_G=current['training_return'], success=current['success'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
    except BaseException:
        save(directory / 'failed.json', dict(traceback=traceback.format_exc(), C=ledger.total, episode=episodes))
        raise
