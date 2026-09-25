"""Run the seven-node QMIX coverage-only (QCO) and coverage-first (QCF) baselines."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))


def main():
    import torch
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['coverage_only', 'coverage_first'], required=True)
    parser.add_argument('--seed', type=int, default=906300)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--budget', type=int, default=20000)
    parser.add_argument('--run', action='store_true', help='Execute training after preflight.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error('Budget must be positive.')
    output = (ROOT / (args.output or Path(f'results/generated/{args.method}_{args.seed}'))).resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output must remain inside this repository.')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib_cache'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from baselines.coverage import runner
    from gevd.environments.prior_graph import PriorGraph
    from gevd.environments.multi_robot import GEVDMultiRobotEnv
    from types import SimpleNamespace

    config_path = ROOT / f'configs/simulation/seven_node/{args.method}.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['environment']['graph_path'] = str(ROOT / config['environment']['graph_path'])
    config['vdn'].update(seed=args.seed, device=args.device)
    config['output']['directory'] = str(output / args.method / f'seed_{args.seed}' / 'run')
    config['search']['total_simulator_budget'] = args.budget
    torch.set_num_threads(1)
    if not args.run:
        e, r = config['environment'], config['reward']
        prior = PriorGraph(e['graph_path'], e['map_width'], e['start_nodes'][0], 'cpu')
        env = GEVDMultiRobotEnv(prior, tuple(e['start_nodes']), e['t_max'], r['alpha'],
            r['beta'], r['rho_v'], r['rho_g'], r['pair_factor_weight'])
        learner = runner.QMIX(SimpleNamespace(config=config, env=env,
            seed=args.seed, device=torch.device(args.device)))
        print(json.dumps({'status': 'preflight_passed', 'method': args.method,
            'network': type(learner.online).__name__, 'nodes': env.num_nodes,
            'intra_weight': 7, 'simulator_calls': 0}))
        return
    if output.exists():
        parser.error('Existing output is protected; select a new --output directory.')
    (output / 'configs').mkdir(parents=True)
    (output / f'configs/{args.method}_{args.seed}.yaml').write_text(
        yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    runner.worker(output, args.method, args.seed, args.budget, args.device)


if __name__ == '__main__':
    main()
