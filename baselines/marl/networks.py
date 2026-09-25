"""Standard QMIX/COMA cores with independent conventional MLP encoders.

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


def tensor(x, device, dtype=torch.float32):
    array = np.asarray(x)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=dtype, device=device)


STANDARD_VERSION = 'standard_common_task_v1'
CHECKPOINT_VERSION = 1
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
    def __init__(self, obs_dim, actions, edges, horizon, prior):
        super().__init__()
        self.edges, self.horizon = edges, horizon
        self.obs_dim = obs_dim
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float32).clone())
        self.net = nn.Sequential(nn.Linear(obs_dim+self.prior.numel(), 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, actions))

    def forward(self, obs):
        normalized = normalized_observations(obs, self.edges, self.horizon)
        return self.net(append_prior(normalized, self.prior))


class BaselineInputs:
    """Common task information/reward; context needs only env/config/device/seed."""
    standard = True
    standalone = True

    def configure_inputs(self, trainer):
        config = trainer.config
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
        self.online = IndependentLocalNetwork(self.o, self.a, e.num_edges,
                                              e.t_max, self.prior).to(self.device)
        self.initial_online_sha256 = network_digest(self.online)

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
            'encoder': 'mlp_128_relu_128_relu',
            'actor_inputs': 'full_env.local_observations + flattened_static_prior_buffer',
            'central_inputs': 'concatenated_full_local_observations + flattened_static_prior_buffer',
            'reward': 'transition.reward', 'optimizer_reward_scale': .1,
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
    def __init__(self, robots, obs_dim, edges, horizon, width=32, prior=()):
        super().__init__()
        self.n, self.o, self.e, self.t, self.h = robots, obs_dim, edges, horizon, width
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float32).clone())
        self.encoder = nn.Sequential(nn.Linear(robots*obs_dim+self.prior.numel(),128),nn.ReLU())
        self.first = nn.Linear(128,robots*width)
        self.second = nn.Linear(128,width)
        self.bias = nn.Linear(128,width)
        self.value = nn.Sequential(nn.Linear(128,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self, utilities, observations):
        normalized=normalized_observations(observations,self.e,self.t)
        z=self.encoder(append_prior(normalized.reshape(-1,self.n*self.o),self.prior))
        w=self.first(z).abs().reshape(-1,self.n,self.h)
        hidden=F.elu(torch.bmm(utilities.unsqueeze(1),w).squeeze(1)+self.bias(z))
        return (hidden*self.second(z).abs()).sum(-1)+self.value(z).squeeze(-1)


class QMIX(BaselineInputs):
    def __init__(self, trainer):
        self.configure_inputs(trainer)
        self.target=copy.deepcopy(self.online)
        e=trainer.env
        self.rng=random.Random(trainer.seed+51)
        self.mixer=MonotonicMixer(self.n,self.o,e.num_edges,e.t_max,prior=self.prior).to(self.device)
        self.target_mixer=copy.deepcopy(self.mixer)
        self.parameters=list(self.online.parameters())+list(self.mixer.parameters())
        self.optimizer=torch.optim.Adam(self.parameters,lr=5e-5)
        self.replay=TransitionReplayBuffer(20000,seed=trainer.seed+50)
        self.updates=0

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
        for r in episode:
            self.replay.push(r['obs'],r['actions'],r['reward'],r['next_obs'],r['mask'],r['next_mask'],r['done'])
        if len(self.replay)<256: return []
        diagnostics=[]
        for _ in range(round(.5*len(episode))):
            b=self.replay.sample(64)
            s=tensor(b.state,self.device); ns=tensor(b.next_state,self.device)
            acts=tensor(b.action,self.device,torch.long)
            mask=tensor(b.next_action_mask,self.device,torch.bool)
            done=tensor(b.done,self.device,torch.bool)
            reward=tensor(b.reward,self.device)*.1
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
            grad=nn.utils.clip_grad_norm_(self.parameters,10.)
            self.optimizer.step(); self.updates+=1
            if self.updates%500==0:
                self.target.load_state_dict(self.online.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            diagnostics.append({'loss':float(loss.detach()),'gradient':float(grad)})
        return diagnostics

    def checkpoint(self):
        return {'format_version':CHECKPOINT_VERSION,'method':'QMIX',
                'online':self.online.state_dict(),'target':self.target.state_dict(),
                'mixer':self.mixer.state_dict(),'target_mixer':self.target_mixer.state_dict(),
                'optimizer':self.optimizer.state_dict(),'updates':self.updates,
                'rng_state':self.rng.getstate(),'replay':self.replay.state_dict(),**self.input_metadata()}

    def load_checkpoint(self, checkpoint):
        self.validate_checkpoint(checkpoint, 'QMIX')
        for name in ('online', 'target', 'mixer', 'target_mixer', 'optimizer', 'replay'):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.rng.setstate(checkpoint['rng_state'])
        self.updates = checkpoint['updates']
        self.initial_online_sha256 = checkpoint['baseline_standard']['initial_online_sha256']


class CounterfactualCritic(nn.Module):
    def __init__(self,n,obs_dim,actions,edges,horizon,prior=()):
        super().__init__()
        self.n,self.o,self.a,self.e,self.t=n,obs_dim,actions,edges,horizon
        self.register_buffer('prior', torch.as_tensor(prior, dtype=torch.float32).clone())
        self.net=nn.Sequential(nn.Linear(n*obs_dim+obs_dim+2*n*actions+n+self.prior.numel(),128),nn.ReLU(),
                               nn.Linear(128,128),nn.ReLU(),nn.Linear(128,actions))
        self.register_buffer('identity',torch.eye(n))

    def forward(self,obs,actions,previous):
        b=obs.shape[0]
        norm=normalized_observations(obs,self.e,self.t)
        global_obs=norm.reshape(b,-1).unsqueeze(1).expand(-1,self.n,-1)
        onehot=F.one_hot(actions,self.a).float()
        others=onehot.unsqueeze(1).expand(-1,self.n,-1,-1)*(1-self.identity).reshape(1,self.n,self.n,1)
        old=previous.reshape(b,1,-1).expand(-1,self.n,-1)
        features=torch.cat([global_obs,norm,others.reshape(b,self.n,-1),old,
                            self.identity.unsqueeze(0).expand(b,-1,-1)],dim=-1)
        return self.net(append_prior(features,self.prior))


def lambda_returns(rewards, next_values, dones, lam=.8):
    """Finite episodes or nonterminal budget prefixes; gamma=1 for this task."""
    result=torch.zeros_like(next_values)
    future=next_values[-1]
    for t in reversed(range(len(rewards))):
        future=rewards[t]+(~dones[t]).float()*((1-lam)*next_values[t]+lam*future)
        result[t]=future
    return result


class COMA(BaselineInputs):
    def __init__(self,trainer):
        self.configure_inputs(trainer)
        e=trainer.env
        self.rng=np.random.default_rng(trainer.seed+71)
        self.critic=CounterfactualCritic(self.n,self.o,self.a,e.num_edges,e.t_max,prior=self.prior).to(self.device)
        self.target=copy.deepcopy(self.critic)
        self.actor_optimizer=torch.optim.RMSprop(self.online.parameters(),lr=.0005,alpha=.99,eps=1e-5)
        self.critic_optimizer=torch.optim.RMSprop(self.critic.parameters(),lr=.0005,alpha=.99,eps=1e-5)
        self.pending=[];self.updates=0;self.actor_updates=0

    def probabilities(self,obs,mask,epsilon):
        if not ((~mask).sum(-1)>0).all():
            raise ValueError('Cannot select an action without legal moves')
        logits=self.online(obs.reshape(-1,self.o)).reshape(-1,self.n,self.a).masked_fill(mask,-torch.inf)
        policy=logits.softmax(-1)
        eps=torch.as_tensor(epsilon,dtype=policy.dtype,device=self.device).reshape(-1,1,1)
        uniform=(~mask).float()/(~mask).sum(-1,keepdim=True)
        return (1-eps)*policy+eps*uniform

    def actions(self,obs,mask,epsilon,exam=False):
        with torch.no_grad():
            pi=self.probabilities(tensor(obs[None],self.device),tensor(mask[None],self.device,torch.bool),0. if exam else epsilon)[0].cpu().numpy()
        if exam:return pi.argmax(-1)
        return np.array([self.rng.choice(self.a,p=p.astype(float)/p.sum(dtype=float)) for p in pi])

    def update(self,episode,force=False):
        self.pending.append(episode)
        if len(self.pending)<8 and not force:return []
        # One on-policy batch: actors unchanged while these episodes were collected.
        critic_diagnostics=[];actor_samples=[]
        for ep in self.pending:
            obs=tensor([r['obs'] for r in ep],self.device)
            acts=tensor([r['actions'] for r in ep],self.device,torch.long)
            masks=tensor([r['mask'] for r in ep],self.device,torch.bool)
            eps=tensor([r['epsilon'] for r in ep],self.device)
            prev=torch.cat([torch.zeros(1,self.n,self.a,device=self.device),F.one_hot(acts[:-1],self.a).float()])
            rewards=tensor([r['reward'] for r in ep],self.device)*.1
            dones=tensor([r['done'] for r in ep],self.device,torch.bool)
            with torch.no_grad():
                taken=self.target(obs,acts,prev).gather(-1,acts.unsqueeze(-1)).squeeze(-1)
                tail=torch.zeros(self.n,device=self.device)
                if not ep[-1]['done']:
                    ns=tensor(ep[-1]['next_obs'][None],self.device)
                    nm=tensor(ep[-1]['next_mask'][None],self.device,torch.bool)
                    pi=self.probabilities(ns,nm,ep[-1]['epsilon'])[0]
                    # At a budget cutoff no factual next action exists: sample one
                    # joint action for the target expectation, without env stepping.
                    probabilities=pi.cpu().numpy()
                    sampled=tensor([self.rng.choice(self.a,p=p.astype(float)/p.sum(dtype=float))
                                    for p in probabilities],self.device,torch.long)
                    tail=self.target(ns,sampled[None],F.one_hot(acts[-1:],self.a).float())[0].gather(-1,sampled[:,None]).squeeze(-1)
                next_values=torch.cat([taken[1:],tail[None]],dim=0)
                targets=lambda_returns(rewards[:,None],next_values,dones[:,None])
            for t in reversed(range(len(ep))):
                q=self.critic(obs[t:t+1],acts[t:t+1],prev[t:t+1])[0]
                loss=(q.gather(-1,acts[t,:,None]).squeeze(-1)-targets[t]).square().mean()
                self.critic_optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),10.)
                self.critic_optimizer.step();self.updates+=1
                critic_diagnostics.append(float(loss.detach()))
            actor_samples.append((obs,acts,masks,eps,prev))
        obs,acts,masks,eps,prev=[torch.cat([r[k] for r in actor_samples]) for k in range(5)]
        with torch.no_grad():q=self.critic(obs,acts,prev)
        pi=self.probabilities(obs,masks,eps)
        advantage=(q.gather(-1,acts.unsqueeze(-1)).squeeze(-1)-(pi.detach()*q).sum(-1)).detach()
        loss=-(advantage*pi.gather(-1,acts.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12).log()).mean()
        self.actor_optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        self.actor_optimizer.step();self.actor_updates+=1
        if self.updates//200 != (self.updates-sum(len(ep) for ep in self.pending))//200:
            self.target.load_state_dict(self.critic.state_dict())
        self.pending=[]
        return [{'actor_loss':float(loss.detach()),'critic_loss':float(np.mean(critic_diagnostics))}]

    def checkpoint(self):
        return {'format_version':CHECKPOINT_VERSION,'method':'COMA',
                'online':self.online.state_dict(),'critic':self.critic.state_dict(),
                'target_critic':self.target.state_dict(),'actor_optimizer':self.actor_optimizer.state_dict(),
                'critic_optimizer':self.critic_optimizer.state_dict(),'critic_updates':self.updates,
                'actor_updates':self.actor_updates,'pending':copy.deepcopy(self.pending),
                'rng_state':copy.deepcopy(self.rng.bit_generator.state),**self.input_metadata()}

    def load_checkpoint(self, checkpoint):
        self.validate_checkpoint(checkpoint, 'COMA')
        for name, key in (('online', 'online'), ('critic', 'critic'),
                          ('target', 'target_critic'), ('actor_optimizer', 'actor_optimizer'),
                          ('critic_optimizer', 'critic_optimizer')):
            getattr(self, name).load_state_dict(checkpoint[key])
        self.rng.bit_generator.state = copy.deepcopy(checkpoint['rng_state'])
        self.pending = copy.deepcopy(checkpoint['pending'])
        self.updates = checkpoint['critic_updates']
        self.actor_updates = checkpoint['actor_updates']
        self.initial_online_sha256 = checkpoint['baseline_standard']['initial_online_sha256']
