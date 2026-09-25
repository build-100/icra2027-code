"""Validated, algorithm-owned settings; GEVD's vdn tuning is not inherited."""
import copy
import math

STANDARD_VERSION = 'standard_common_task_v2'
R8_VERSION = 'task8_success_rank_arctan_delta_v1'

COMMON = dict(hidden_dim=128, central_hidden_dim=128, gradient_clip=10.0,
              td_reward_scale=0.1, gamma=1.0, initialization='pytorch',
              epsilon_decay_basis='factual_steps', epsilon_decay_steps=6000,
              epsilon_decay_episodes=1000)
DEFAULTS = {
    'qmix': dict(COMMON, learning_rate=5e-5, batch_size=64, replay_capacity=20000,
                 min_replay=256, updates_per_factual_step=0.5,
                 target_update_interval=500, mixer_width=32,
                 epsilon_start=1.0, epsilon_end=0.05),

}


def resolve_settings(config, method):
    settings = copy.deepcopy(DEFAULTS[method])
    supplied = config.get('baseline', {})
    unknown = set(supplied) - set(settings)
    if unknown:
        raise ValueError(f'Unknown {method} baseline settings: {sorted(unknown)}')
    settings.update(supplied)
    integer_fields = {'hidden_dim', 'central_hidden_dim', 'batch_size', 'replay_capacity',
                      'min_replay', 'target_update_interval', 'mixer_width',
                      'batch_episodes', 'epsilon_decay_steps', 'epsilon_decay_episodes'}
    unit_fields = {'epsilon_start', 'epsilon_end', 'td_lambda', 'rmsprop_alpha'}
    for key, value in settings.items():
        if key in ('initialization', 'epsilon_decay_basis'):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'baseline.{key} must be a finite number')
        if key in integer_fields and (not isinstance(value, int) or value < 1):
            raise ValueError(f'baseline.{key} must be a positive integer')
        if key in unit_fields:
            if not 0 <= value <= 1:
                raise ValueError(f'baseline.{key} must be in [0, 1]')
        elif key == 'updates_per_factual_step':
            if value < 0:
                raise ValueError('updates_per_factual_step must be nonnegative')
        elif value <= 0:
            raise ValueError(f'baseline.{key} must be positive')
    if settings['gamma'] != 1.0:
        raise ValueError('This finite-horizon task requires gamma=1')
    if settings['epsilon_end'] > settings['epsilon_start']:
        raise ValueError('epsilon_end must not exceed epsilon_start')
    if settings['epsilon_decay_basis'] not in ('factual_steps', 'episodes'):
        raise ValueError('epsilon_decay_basis must be factual_steps or episodes')
    if settings['initialization'] not in ('pytorch', 'orthogonal'):
        raise ValueError('initialization must be pytorch or orthogonal')
    if method == 'qmix' and not settings['batch_size'] <= settings['min_replay'] <= settings['replay_capacity']:
        raise ValueError('Require batch_size <= min_replay <= replay_capacity')
    return settings
