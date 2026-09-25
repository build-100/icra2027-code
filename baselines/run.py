"""Run the independent QMIX/COMA learners or CMRE/sGre/dGre planners."""
from pathlib import Path
import argparse
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['qmix', 'coma', 'cmre', 'sgre', 'dgre'])
    parser.add_argument('--map', default='map8', choices=['map3', 'map7', 'map8'])
    parser.add_argument('--robots', type=int, default=3, choices=[2, 3, 4])
    parser.add_argument('--seed', type=int, default=906300)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--run', action='store_true', help='Execute training or planning after preflight.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (ROOT / (args.output or Path(
        f'results/generated/{args.map}_N{args.robots}_{args.method}_{args.seed}'))).resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output must remain inside this repository.')
    config_path = ROOT / f'configs/simulation/comparison/{args.map}/N{args.robots}/{args.method}.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['environment']['graph_path'] = str(ROOT / config['environment']['graph_path'])
    config['vdn'].update(seed=args.seed, device=args.device)
    if args.budget is not None:
        config['search']['total_simulator_budget'] = args.budget
    budget = config['search']['total_simulator_budget']
    if budget <= 0:
        parser.error('Budget must be positive.')
    config['output']['directory'] = str(output)
    if args.method in ('qmix', 'coma'):
        from baselines.marl import runner
        QMIX, COMA, make_context, _ = runner.setup()
        context = make_context(config)
        learner = (QMIX if args.method == 'qmix' else COMA)(context)
        if not args.run:
            print(json.dumps({'status': 'preflight_passed', 'method': args.method,
                'network': type(learner.online).__name__, 'nodes': context.env.num_nodes,
                'simulator_calls': 0}))
            return
        if output.exists():
            parser.error('Existing output is protected; select a new --output directory.')
        output.mkdir(parents=True)
        effective_config = output / 'config.yaml'
        effective_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        runner.run_one({'method': args.method, 'map': args.map, 'N': args.robots,
            'seed': args.seed, 'config': str(effective_config), 'directory': str(output)})
        return

    from gevd.environments.prior_graph import PriorGraph
    from gevd.environments.multi_robot import GEVDMultiRobotEnv
    from baselines.planning.planners import CoordinatedPlanner
    from baselines.planning.double_greedy import CGEDoubleGreedyPlanner

    environment, reward = config['environment'], config['reward']

    def make_environment():
        prior = PriorGraph(environment['graph_path'], environment['map_width'],
            environment['start_nodes'][0], 'cpu')
        return GEVDMultiRobotEnv(prior, environment['start_nodes'], environment['t_max'],
            reward['alpha'], reward['beta'], reward['rho_v'], reward['rho_g'],
            reward['pair_factor_weight'])

    if budget <= environment['t_max']:
        parser.error('Planning budget must exceed the horizon to reserve final validation.')
    planning_budget = budget - environment['t_max']
    planner = (CGEDoubleGreedyPlanner(make_environment(), planning_budget, args.seed, 32)
        if args.method == 'dgre' else CoordinatedPlanner(make_environment(),
            'coverage_only' if args.method == 'cmre' else 'cge_adapted', planning_budget, args.seed))
    if not args.run:
        print(json.dumps({'status': 'preflight_passed', 'method': args.method,
            'nodes': planner.env.num_nodes, 'simulator_calls': 0}))
        return
    if output.exists():
        parser.error('Existing output is protected; select a new --output directory.')
    output.mkdir(parents=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    result = planner.run()
    validation_calls = 0
    if result['selected']:
        checker = CoordinatedPlanner(make_environment(), 'coverage_only', environment['t_max'], args.seed)
        actual = checker.evaluate(result['selected']['routes'])
        validation_calls = checker.used
        assert actual['success'] and math.isclose(actual['S'], result['selected']['S'], abs_tol=1e-8)
    result.update(total_calls=planner.used + validation_calls,
        counts={'planner': planner.used, 'candidate_validation': validation_calls})
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'completed': True, 'calls': result['total_calls'],
        'success': bool(result['selected'])}))


if __name__ == '__main__':
    main()
