"""Seven-node QMIX core with an independent conventional MLP encoder.

Actors receive full environment observations (including cumulative history and
broadcasts) and the raw static prior, flattened with ordinary normalization.
Central networks receive the same prior and concatenated local observations.
The task reward is unchanged; the existing optimizer scale is 0.1. This is a
task adaptation, not a layer-for-layer SMAC recurrent or paper-score reproduction.
Historical graph/shared and restricted-input implementations require frozen code.
"""
import copy
import hashlib
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from baselines.replay import TransitionReplayBuffer
from .settings import STANDARD_VERSION, resolve_settings


def tensor(x, device, dtype=torch.float32):
    array = np.asarray(x)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=dtype, device=device)


CHECKPOINT_VERSION = 2
PRIOR_FIELDS = ('edge_endpoints', 'edge_normalized_weights', 'edge_distances',
                'neighbor_node_indices', 'neighbor_edge_indices')


def static_prior(env):
    """Invertible field normalization; no graph processing or learned features."""
    parts, layout = [], {}
    cursor = 0
    for name in PRIOR_FIELDS:
        raw = np.asarray(getattr(env, name), dtype=np.float32)
        if name in ('edge_normalized_weights', 'edge_distances'):
            normalized = np.sign(raw)*np.log1p(np.abs(raw))
            rule = 'signed_log1p'
        else:
            scale = max(1, (env.num_edges if name == 'neighbor_edge_indices'
                            else env.num_nodes)-1)
            normalized = raw/scale
            rule = f'divide_by_{scale}; negative_padding_preserved'
        parts.append(normalized.ravel())
        layout[name] = {'shape': list(raw.shape), 'slice': [cursor, cursor+raw.size],
                        'normalization': rule}
        cursor += raw.size
    return np.concatenate(parts), layout


def normalized_observations(obs, edges, horizon):
    normalized = obs.clone()
    normalized[..., :edges] /= horizon
    normalized[..., -1] /= horizon
    return normalized


def append_prior(features, prior):
    return torch.cat((features, prior.expand(*features.shape[:-1], -1)), dim=-1)


def network_digest(network):
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class IndependentLocalNetwork(nn.Module):
    """Plain MLP, independent of GEVD; static prior is a nontrainable buffer."""
    def __init__(self, obs_dim, actions, edges, horizon, prior, hidden_dim=128):
        super().__init__()
        self.edges, self.horizon = edges, horizon
        self.obs_dim = obs_dim
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float32).clone())
        self.net = nn.Sequential(nn.Linear(obs_dim+self.prior.numel(), hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, actions))

    def forward(self, obs):
        normalized = normalized_observations(obs, self.edges, self.horizon)
        return self.net(append_prior(normalized, self.prior))


class BaselineInputs:
    """Common task information/reward; context needs only env/config/device/seed."""
    standard = True
    standalone = True

    def configure_inputs(self, trainer):
        config = trainer.config
        self.settings = resolve_settings(config, self.__class__.__name__.lower())
        self.completed_episodes = 0
        if 'baseline_independent' in config:
            raise ValueError('baseline_independent is historical: use its frozen source; '
                             'standard baselines do not reinterpret old configurations')
        if config.get('event_aligned', {}).get('enabled', False):
            raise ValueError('Standard baselines do not accept event_aligned credit')
        mcbr = config.get('mcbr', {})
        if (mcbr.get('enabled', False) or mcbr.get('branch_roots_per_episode', 0)
                or mcbr.get('branch_fraction', 0)):
            raise ValueError('Standard baselines do not accept MCBR')
        for name in ('policy_retention', 'exact_terminal'):
            if config.get(name, {}).get('enabled', False):
                raise ValueError(f'Standard baselines do not accept {name}')
        self.device = trainer.device
        e = trainer.env
        self.n, self.a = e.num_robots, e.action_dim
        self.o = e.observation_dim
        self.prior, self.prior_layout = static_prior(e)
        self.observation_layout = copy.deepcopy(e.observation_layout)
        self.horizon = e.t_max
        self.environment_schema = copy.deepcopy(e.schema())
        self.environment_schema['objective_version'] = (
            'task8_success_rank_arctan_delta_v1' if hasattr(e, 'rank_tau') else 'standard_environment_reward_v2')
        if hasattr(e, 'rank_tau'):
            self.environment_schema['reward_parameters']['rank_tau'] = e.rank_tau
        self.online = IndependentLocalNetwork(self.o, self.a, e.num_edges,
                                              e.t_max, self.prior, self.settings['hidden_dim']).to(self.device)
        if self.settings['initialization'] == 'orthogonal':
            def initialize(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=2**0.5)
                    nn.init.zeros_(module.bias)
            self.online.apply(initialize)
            nn.init.orthogonal_(self.online.net[-1].weight, gain=1.0)
        self.initial_online_sha256 = network_digest(self.online)

    def epsilon(self, factual_steps, exam=False):
        if exam:
            return 0.0
        cfg = self.settings
        episodes = cfg['epsilon_decay_basis'] == 'episodes'
        progress = self.completed_episodes if episodes else factual_steps
        duration = cfg['epsilon_decay_episodes'] if episodes else cfg['epsilon_decay_steps']
        fraction = min(max(progress / duration, 0.0), 1.0)
        return cfg['epsilon_start'] + fraction * (cfg['epsilon_end'] - cfg['epsilon_start'])

    def record_episode(self, episode):
        if episode and episode[-1]['done']:
            self.completed_episodes += 1

    def observations(self, env):
        return env.local_observations()

    def training_reward(self, env, transition):
        return transition.reward

    def input_metadata(self):
        return {'baseline_standard': {'enabled': True, 'version': STANDARD_VERSION,
            'observation_dim': self.o, 'observation_fields': list(self.observation_layout),
            'observation_layout': copy.deepcopy(self.observation_layout),
            'dynamic_normalization': 'self_traversals and time divided by t_max',
            't_max': self.horizon, 'robots': self.n, 'actions': self.a,
            'static_prior_dim': len(self.prior), 'static_prior_fields': copy.deepcopy(self.prior_layout),
            'static_prior_sha256': hashlib.sha256(self.prior.tobytes()).hexdigest(),
            'actor_input_dim': self.o+len(self.prior),
            'encoder': f"mlp_{self.settings['hidden_dim']}_relu_{self.settings['hidden_dim']}_relu",
            'actor_inputs': 'full_env.local_observations + flattened_static_prior_buffer',
            'central_inputs': 'concatenated_full_local_observations + flattened_static_prior_buffer',
            'reward': 'transition.reward', 'optimizer_reward_scale': self.settings['td_reward_scale'],
            'settings': copy.deepcopy(self.settings), 'environment_schema': self.environment_schema,
            'execution_parameters': sum(p.numel() for p in self.online.parameters()),
            'central_parameters': sum(p.numel() for p in
                (self.mixer if hasattr(self, 'mixer') else self.critic).parameters()),
            'paper_reproduction': False, 'recurrent_encoder': False,
            'hash_scope': 'independent_online_mlp_state_dict_including_static_prior',
            'initial_online_sha256': self.initial_online_sha256}}

    def validate_checkpoint(self, checkpoint, method):
        if (checkpoint.get('format_version') != CHECKPOINT_VERSION
                or checkpoint.get('method') != method
                or 'baseline_standard' not in checkpoint
                or 'baseline_independent' in checkpoint):
            raise ValueError('Legacy or incompatible standard baseline checkpoint')
        expected = self.input_metadata()['baseline_standard']
        actual = checkpoint['baseline_standard']
        if any(actual.get(k) != v for k, v in expected.items() if k != 'initial_online_sha256'):
            raise ValueError('Standard baseline checkpoint input schema/prior mismatch')


class MonotonicMixer(nn.Module):
    def __init__(self, robots, obs_dim, edges, horizon, width=32, prior=(), hidden_dim=128):
        super().__init__()
        self.n, self.o, self.e, self.t, self.h = robots, obs_dim, edges, horizon, width
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float32).clone())
        self.encoder = nn.Sequential(nn.Linear(robots*obs_dim+self.prior.numel(),hidden_dim),nn.ReLU())
        self.first = nn.Linear(hidden_dim,robots*width)
        self.second = nn.Linear(hidden_dim,width)
        self.bias = nn.Linear(hidden_dim,width)
        self.value = nn.Sequential(nn.Linear(hidden_dim,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self, utilities, observations):
        normalized=normalized_observations(observations,self.e,self.t)
        z=self.encoder(append_prior(normalized.reshape(-1,self.n*self.o),self.prior))
        w=self.first(z).abs().reshape(-1,self.n,self.h)
        hidden=F.elu(torch.bmm(utilities.unsqueeze(1),w).squeeze(1)+self.bias(z))
        return (hidden*self.second(z).abs()).sum(-1)+self.value(z).squeeze(-1)


class QMIX(BaselineInputs):
    def __init__(self, trainer):
        self.configure_inputs(trainer)
        cfg = self.settings
        self.target=copy.deepcopy(self.online)
        e=trainer.env
        self.rng=random.Random(trainer.seed+51)
        self.mixer=MonotonicMixer(self.n,self.o,e.num_edges,e.t_max,prior=self.prior,
            width=cfg['mixer_width'],hidden_dim=cfg['central_hidden_dim']).to(self.device)
        self.target_mixer=copy.deepcopy(self.mixer)
        self.parameters=list(self.online.parameters())+list(self.mixer.parameters())
        self.optimizer=torch.optim.Adam(self.parameters,lr=cfg['learning_rate'])
        self.replay=TransitionReplayBuffer(cfg['replay_capacity'],seed=trainer.seed+50)
        self.updates=0
        self.update_credit=0.0

    def actions(self, obs, mask, epsilon, exam=False):
        with torch.no_grad():
            q=self.online(tensor(obs,self.device)).masked_fill(
                tensor(mask,self.device,torch.bool),-torch.inf)
            actions=q.argmax(-1).cpu().numpy()
        for i in range(self.n):
            legal=np.flatnonzero(~mask[i])
            if not len(legal):
                raise ValueError('Cannot select an action without legal moves')
            if not exam and self.rng.random()<epsilon:
                actions[i]=self.rng.choice(legal.tolist())
        return actions

    def update(self, episode):
        self.record_episode(episode)
        eligible_steps = 0
        for r in episode:
            self.replay.push(r['obs'],r['actions'],r['reward'],r['next_obs'],r['mask'],r['next_mask'],r['done'])
            eligible_steps += len(self.replay) >= self.settings['min_replay']
        # Warmup transitions do not create deferred optimization debt. The step
        # reaching min_replay is eligible; fractional credit survives episodes.
        self.update_credit += self.settings['updates_per_factual_step'] * eligible_steps
        if len(self.replay)<self.settings['min_replay']: return []
        diagnostics=[]
        count = int(self.update_credit + 1e-12)
        self.update_credit = max(0.0, self.update_credit - count)
        for _ in range(count):
            b=self.replay.sample(self.settings['batch_size'])
            s=tensor(b.state,self.device); ns=tensor(b.next_state,self.device)
            acts=tensor(b.action,self.device,torch.long)
            mask=tensor(b.next_action_mask,self.device,torch.bool)
            done=tensor(b.done,self.device,torch.bool)
            reward=tensor(b.reward,self.device)*self.settings['td_reward_scale']
            local=self.online(s.reshape(-1,self.o)).reshape(-1,self.n,self.a)
            pred=self.mixer(local.gather(-1,acts.unsqueeze(-1)).squeeze(-1),s)
            with torch.no_grad():
                targets=reward.clone(); active=~done
                if active.any():
                    sm=mask[active]
                    assert ((~sm).sum(-1)>0).all()
                    choice=self.online(ns[active].reshape(-1,self.o)).reshape(-1,self.n,self.a).masked_fill(sm,-torch.inf).argmax(-1)
                    values=self.target(ns[active].reshape(-1,self.o)).reshape(-1,self.n,self.a).gather(-1,choice.unsqueeze(-1)).squeeze(-1)
                    targets[active]+=self.target_mixer(values,ns[active])
            loss=(pred-targets).square().mean()
            self.optimizer.zero_grad();loss.backward()
            grad=nn.utils.clip_grad_norm_(self.parameters,self.settings['gradient_clip'])
            self.optimizer.step(); self.updates+=1
            if self.updates%self.settings['target_update_interval']==0:
                self.target.load_state_dict(self.online.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            diagnostics.append({'loss':float(loss.detach()),'gradient':float(grad)})
        return diagnostics

    def checkpoint(self):
        return {'format_version':CHECKPOINT_VERSION,'method':'QMIX',
                'online':self.online.state_dict(),'target':self.target.state_dict(),
                'mixer':self.mixer.state_dict(),'target_mixer':self.target_mixer.state_dict(),
                'optimizer':self.optimizer.state_dict(),'updates':self.updates,
                'update_credit':self.update_credit,'completed_episodes':self.completed_episodes,
                'rng_state':self.rng.getstate(),'replay':self.replay.state_dict(),**self.input_metadata()}

    def load_checkpoint(self, checkpoint):
        self.validate_checkpoint(checkpoint, 'QMIX')
        for name in ('online', 'target', 'mixer', 'target_mixer', 'optimizer', 'replay'):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.rng.setstate(checkpoint['rng_state'])
        self.updates = checkpoint['updates']
        self.update_credit = checkpoint['update_credit']
        self.completed_episodes = checkpoint['completed_episodes']
        self.initial_online_sha256 = checkpoint['baseline_standard']['initial_online_sha256']
