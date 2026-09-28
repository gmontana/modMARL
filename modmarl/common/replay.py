"""Typed replay storage for the data contracts used across modMARL.

Model: preallocated NumPy ring buffers own transitions or padded whole episodes and emit
typed Torch batches on a requested device. Invariants: agent ordering and algorithm-specific
state—relations, shared memory, media, and padding masks—are stored explicitly; termination
is distinct from time-limit padding. Interface: each buffer exposes ``add``/``add_episode``,
``sample``, and length without hiding an algorithm-specific reshape in the learner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class MARCReplayBatch:
    obs: torch.Tensor
    node_features: torch.Tensor
    relations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_node_features: torch.Tensor
    next_relations: torch.Tensor
    dones: torch.Tensor


class MARCReplayBuffer:
    """Replay buffer for MARC: also stores the per-step relational graph (node features + relations)."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        n_entities: int,
        node_feature_dim: int,
        num_relations: int,
    ) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.node_features = np.zeros((capacity, n_agents, n_entities, node_feature_dim), dtype=np.float32)
        self.relations = np.zeros((capacity, num_relations, n_entities, n_entities), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_node_features = np.zeros((capacity, n_agents, n_entities, node_feature_dim), dtype=np.float32)
        self.next_relations = np.zeros((capacity, num_relations, n_entities, n_entities), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        node_features: np.ndarray,
        relations: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_node_features: np.ndarray,
        next_relations: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.node_features[self.ptr] = node_features
        self.relations[self.ptr] = relations
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_node_features[self.ptr] = next_node_features
        self.next_relations[self.ptr] = next_relations
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device, *, normalize_rewards: bool = False,
    ) -> MARCReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        rewards = self.rewards[indices]
        if normalize_rewards:
            # Released `norm_rews: true` (utils/buffer.py): standardise against the
            # statistics of the filled buffer, not of the sampled minibatch.
            filled = self.rewards[: self.size]
            rewards = (rewards - filled.mean()) / (filled.std() + 1e-8)
        return MARCReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            node_features=torch.as_tensor(self.node_features[indices], device=device),
            relations=torch.as_tensor(self.relations[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(rewards, device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            next_node_features=torch.as_tensor(self.next_node_features[indices], device=device),
            next_relations=torch.as_tensor(self.next_relations[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class MemoryReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    memory_seen: torch.Tensor      # (batch, n_agents, memory_dim): memory each agent read
    memory_written: torch.Tensor   # (batch, n_agents, memory_dim): memory each agent wrote


class MemoryReplayBuffer:
    """Replay buffer for MD-MADDPG: also stores the shared memory each agent read and wrote."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, memory_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.memory_seen = np.zeros((capacity, n_agents, memory_dim), dtype=np.float32)
        self.memory_written = np.zeros((capacity, n_agents, memory_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        memory_seen: np.ndarray,
        memory_written: np.ndarray,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.memory_seen[self.ptr] = memory_seen
        self.memory_written[self.ptr] = memory_written
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> MemoryReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return MemoryReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
            memory_seen=torch.as_tensor(self.memory_seen[indices], device=device),
            memory_written=torch.as_tensor(self.memory_written[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class MADDPGMReplayBatch:
    obs: torch.Tensor
    comm_actions: torch.Tensor
    medium: torch.Tensor
    actions: torch.Tensor
    ext_rewards: torch.Tensor
    int_rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class MADDPGMReplayBuffer:
    """Replay buffer for MADDPG-M: also stores the broadcast willingnesses, the shared medium,
    and the extrinsic + intrinsic rewards that train the two policy levels."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int = 1) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.comm_actions = np.zeros((capacity, n_agents), dtype=np.float32)
        self.medium = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.ext_rewards = np.zeros((capacity,), dtype=np.float32)
        self.int_rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        comm_actions: np.ndarray,
        medium: np.ndarray,
        actions: np.ndarray,
        ext_reward: float,
        int_reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.comm_actions[self.ptr] = comm_actions
        self.medium[self.ptr] = medium
        self.actions[self.ptr] = actions
        self.ext_rewards[self.ptr] = ext_reward
        self.int_rewards[self.ptr] = int_reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> MADDPGMReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return MADDPGMReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            comm_actions=torch.as_tensor(self.comm_actions[indices], device=device),
            medium=torch.as_tensor(self.medium[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            ext_rewards=torch.as_tensor(self.ext_rewards[indices], device=device),
            int_rewards=torch.as_tensor(self.int_rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class EpisodeBatch:
    obs: torch.Tensor       # (B, T+1, n_agents, obs_dim) — includes the final observation
    actions: torch.Tensor   # (B, T, n_agents) long
    rewards: torch.Tensor   # (B, T) shared team reward
    dones: torch.Tensor     # (B, T): 1.0 where that step terminated the episode
    mask: torch.Tensor      # (B, T): 1.0 for real steps, 0.0 for padding after episode end


class EpisodeReplayBuffer:
    """Whole-episode replay for recurrent Q-learners (the PyMARL pattern).

    Episodes shorter than the horizon are zero-padded and ``mask`` marks the real
    steps, so time-limit truncation and padding are distinct from termination
    (``dones``). Stores only observations/actions/rewards — recurrent learners
    re-run the forward pass (resampling any stochastic messages) at train time.
    """

    def __init__(self, capacity: int, horizon: int, n_agents: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.horizon = horizon
        self.obs = np.zeros((capacity, horizon + 1, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, horizon, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, horizon), dtype=np.float32)
        self.dones = np.zeros((capacity, horizon), dtype=np.float32)
        self.mask = np.zeros((capacity, horizon), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add_episode(
        self,
        *,
        obs: np.ndarray,       # (L+1, n_agents, obs_dim)
        actions: np.ndarray,   # (L, n_agents)
        rewards: np.ndarray,   # (L,)
        dones: np.ndarray,     # (L,)
    ) -> None:
        length = actions.shape[0]
        if length > self.horizon:
            raise ValueError(f"episode length {length} exceeds horizon {self.horizon}")
        for field in (self.obs, self.actions, self.rewards, self.dones, self.mask):
            field[self.ptr] = 0
        self.obs[self.ptr, : length + 1] = obs
        self.actions[self.ptr, :length] = actions
        self.rewards[self.ptr, :length] = rewards
        self.dones[self.ptr, :length] = dones
        self.mask[self.ptr, :length] = 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> EpisodeBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._gather(indices, device)

    def sample_latest(self, batch_size: int, device: torch.device) -> EpisodeBatch:
        """The most recently added ``batch_size`` episodes, oldest first — the
        on-policy batch for actor-critic learners on this scaffold."""
        count = min(batch_size, self.size)
        indices = (self.ptr - count + np.arange(count)) % self.capacity
        return self._gather(indices, device)

    def _gather(self, indices: np.ndarray, device: torch.device) -> EpisodeBatch:
        return EpisodeBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
            mask=torch.as_tensor(self.mask[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size
