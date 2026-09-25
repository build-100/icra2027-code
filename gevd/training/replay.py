from collections import namedtuple
import random
import numpy as np


TRANSITION = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask', 'prob', 'done'))

DQNTransition = namedtuple(
    "DQNTransition",
    (
        "state",
        "action",
        "reward",
        "next_state",
        "action_mask",
        "next_action_mask",
        "done",
        "reward_channels",
    ),
)
DQN_TRANSITION = DQNTransition


class TransitionReplayBuffer:
    """Uniform replay over individual DQN transitions."""

    def __init__(self, capacity, seed=0):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0
        self.rng = random.Random(seed)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        action_mask,
        next_action_mask,
        done,
        reward_channels=None,
    ):
        action_array = np.asarray(action, dtype=np.int64)
        action_value = (
            int(action_array.item()) if action_array.ndim == 0 else action_array.copy()
        )
        transition = DQN_TRANSITION(
            np.asarray(state, dtype=np.float32).copy(),
            action_value,
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            np.asarray(action_mask, dtype=np.bool_).copy(),
            np.asarray(next_action_mask, dtype=np.bool_).copy(),
            bool(done),
            {} if reward_channels is None else dict(reward_channels),
        )
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if batch_size > len(self.memory):
            raise ValueError("Cannot sample more transitions than the replay contains.")
        return transitions_to_batch(self.sample_items(batch_size))

    def sample_items(self, batch_size):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if batch_size > len(self.memory):
            raise ValueError("Cannot sample more transitions than the replay contains.")
        return self.rng.sample(self.memory, batch_size)

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "memory": self.memory[:],
            "position": self.position,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        capacity = int(state["capacity"])
        memory = list(state["memory"])
        if capacity <= 0 or len(memory) > capacity:
            raise ValueError("Invalid replay state.")
        position = int(state["position"])
        if position < 0 or position >= capacity:
            raise ValueError("Invalid replay write position.")
        self.capacity = capacity
        self.memory = memory
        self.position = position
        self.rng.setstate(state["rng_state"])

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)


def transitions_to_batch(transitions):
    transitions = list(transitions)
    if not transitions:
        raise ValueError("Cannot form an empty transition batch.")
    return DQN_TRANSITION(*zip(*transitions))


class MixedReplaySampler:
    """Fixed-fraction factual/branch sampling over two uniform replay buffers."""

    def __init__(self, branch_fraction, seed=0):
        branch_fraction = float(branch_fraction)
        if not 0.0 <= branch_fraction <= 1.0:
            raise ValueError("branch_fraction must be in [0, 1].")
        self.branch_fraction = branch_fraction
        self.rng = random.Random(int(seed))
        self.last_counts = {"factual": 0, "branch": 0}

    def requested_branch_count(self, batch_size):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        # Explicit implementation choice: deterministic round-half-up.
        return min(
            int(batch_size),
            max(0, int(np.floor(batch_size * self.branch_fraction + 0.5))),
        )

    def sample(self, factual_buffer, branch_buffer, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if len(branch_buffer) == 0 or self.branch_fraction == 0.0:
            branch_count = 0
        else:
            branch_count = min(
                self.requested_branch_count(batch_size), len(branch_buffer)
            )
        factual_count = batch_size - branch_count
        if len(factual_buffer) < factual_count:
            raise ValueError(
                "Factual replay is too small for the requested fixed branch fraction."
            )
        factual_items = (
            factual_buffer.sample_items(factual_count) if factual_count else []
        )
        branch_items = (
            branch_buffer.sample_items(branch_count) if branch_count else []
        )
        combined = factual_items + branch_items
        self.rng.shuffle(combined)
        self.last_counts = {"factual": factual_count, "branch": branch_count}
        return transitions_to_batch(combined), dict(self.last_counts)

    def state_dict(self):
        return {
            "branch_fraction": self.branch_fraction,
            "rng_state": self.rng.getstate(),
            "last_counts": dict(self.last_counts),
        }

    def load_state_dict(self, state):
        branch_fraction = float(state["branch_fraction"])
        if branch_fraction != self.branch_fraction:
            raise ValueError("Replay-mixture branch fraction does not match config.")
        self.rng.setstate(state["rng_state"])
        self.last_counts = dict(state.get("last_counts", self.last_counts))


class EpisodeBuffer:
    def __init__(self):
        self.memory = []
        self.Transition = TRANSITION
        self.position = 0

    def push(self, state, action, next_state, reward, mask, prob, done):
        self.memory.append(None)
        self.memory[self.position] = self.Transition(state, action, next_state, reward, mask, prob, done)
        self.position = self.position + 1

    def sample(self):
        transitions = self.memory[:]
        batch = self.Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class EpisodeReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.memory, batch_size)
        return sampled_episodes

    def clear(self):
        self.position = 0
        del self.memory[:]

    def __len__(self):
        return len(self.memory)


class PrioritizedEpisodeReplayMemory:
    def __init__(self, capacity, prob_alpha=0.6):
        self.capacity = capacity
        self.prob_alpha = prob_alpha
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, episode):
        max_prio = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.prob_alpha
        probs /= np.sum(probs)
        indices = np.random.choice(len(self.memory), batch_size, p=probs if np.nansum(probs) == 1 else None)
        sampled_episodes = [self.memory[idx] for idx in indices]

        # sampled_episodes = random.sample(self.memory, batch_size)

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        return sampled_episodes, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def clear(self):
        self.position = 0
        del self.memory[:]

    def __len__(self):
        return len(self.memory)
