"""Shared task environment for standard MARL, without constructing GEVD learners."""
from pathlib import Path
from types import SimpleNamespace
import copy
import random
import numpy as np
import torch
from gevd.environments.prior_graph import PriorGraph
from gevd.environments.multi_robot import GEVDMultiRobotEnv

ROOT = Path(__file__).resolve().parents[2]


def standard_config(config, method):
    if method not in ('qmix', 'coma'):
        raise ValueError('Standard MARL method must be qmix or coma')
    cfg = copy.deepcopy(config)
    if 'baseline_independent' in cfg:
        raise ValueError('Historical restricted baseline config: use its frozen source, not standard MARL')
    cfg.setdefault('event_aligned', {})['enabled'] = False
    cfg.setdefault('mcbr', {}).update(branch_roots_per_episode=0, branch_fraction=0.)
    if 'enabled' in cfg['mcbr']:
        cfg['mcbr']['enabled'] = False
    cfg.setdefault('policy_retention', {})['enabled'] = False
    if 'exact_terminal' in cfg:
        cfg['exact_terminal']['enabled'] = False
    cfg.setdefault('experiment', {}).update(method=method, baseline_version='standard_common_task_v1')
    return cfg


def make_context(config):
    """Only env/config/device/seed; no VDN, graph utility net, credit or MCBR object."""
    cfg = copy.deepcopy(config)
    if cfg.get('event_aligned', {}).get('enabled', False):
        raise ValueError('Standard MARL does not accept GEVD event credit')
    if (cfg.get('mcbr', {}).get('enabled', False) or cfg.get('mcbr', {}).get('branch_roots_per_episode', 0)
            or cfg.get('mcbr', {}).get('branch_fraction', 0)):
        raise ValueError('Standard MARL does not accept MCBR')
    if cfg.get('policy_retention', {}).get('enabled', False) or cfg.get('exact_terminal', {}).get('enabled', False):
        raise ValueError('Standard MARL does not accept GEVD auxiliary modules')
    if 'baseline_independent' in cfg:
        raise ValueError('Historical restricted baseline config requires its frozen source')
    seed = int(cfg['vdn']['seed'])
    device = torch.device(cfg['vdn'].get('device', 'cpu'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    e, r = cfg['environment'], cfg['reward']
    prior = PriorGraph(str(ROOT/e['graph_path']), float(e['map_width']), e['start_nodes'][0], device)
    env = GEVDMultiRobotEnv(prior, tuple(e['start_nodes']), int(e['t_max']), r['alpha'],
        float(r['beta']), float(r['rho_v']), float(r['rho_g']), r['pair_factor_weight'])
    return SimpleNamespace(env=env, config=cfg, device=device, seed=seed)
