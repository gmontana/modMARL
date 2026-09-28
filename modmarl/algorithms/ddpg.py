"""Deep Deterministic Policy Gradient for bounded continuous control.

Model: one actor maps a local observation to a bounded continuous action; one local
critic evaluates that observation/action pair. Replay and target copies make the
one-step deterministic policy-gradient update off-policy.
Invariants: actions stay continuous through acting, replay, targets, and policy
gradients; exploration is temporally correlated Ornstein--Uhlenbeck noise; target
parameters use Polyak updates and never receive gradients.
Interface: :class:`DDPGAgent` owns acting and learning, while :class:`DDPGConfig`
records the paper's experimental schedule.
Why: Lillicrap et al. define DDPG for continuous actions. A categorical Gumbel policy
is a different algorithm, so discrete multi-agent environments are not accepted by
this implementation.

Paper: Lillicrap et al., "Continuous Control with Deep Reinforcement Learning",
ICLR 2016, arXiv:1509.02971v6. Architecture and defaults follow Sections 3 and 7:
400/300-unit actor and critic, action injection at the critic's second layer,
fanin initialization with a 3e-3 final range, Adam at 1e-4/1e-3, critic L2 1e-2,
gamma 0.99, tau 0.001, batch 64, replay capacity 1e6, and OU exploration with
theta 0.15 and sigma 0.2. Low-dimensional networks batch-normalize the state
input, every actor hidden layer, and the critic layers before action injection;
exploration/evaluation use their running statistics as specified in Section 3.
OU sigma is measured directly in environment action units, as in the paper.
Validation uses the continuous fixed-broadcasting navigation environment; each
independent learner receives the cooperative reward.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..components import soft_update_module


@dataclass(frozen=True)
class DDPGUpdate:
    """Scalar diagnostics returned by one deterministic policy-gradient update."""

    critic_loss: float
    actor_loss: float
    target_q: float


@dataclass(frozen=True)
class DDPGConfig:
    """Experimental constants from Lillicrap et al. Sections 3 and 7."""

    batch_size: int = 64
    replay_capacity: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.001
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    critic_weight_decay: float = 1e-2
    ou_theta: float = 0.15
    ou_sigma: float = 0.2

    def update_due(self, replay_size: int) -> bool:
        """Return whether replay contains one complete paper-sized minibatch."""
        return replay_size >= self.batch_size


@dataclass(frozen=True)
class DDPGReplayBatch:
    """One continuous-action replay minibatch."""

    obs: Tensor
    actions: Tensor
    rewards: Tensor
    next_obs: Tensor
    dones: Tensor


class DDPGReplayBuffer:
    """Fixed-capacity replay retaining continuous actions without quantization."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pointer = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.pointer] = obs
        self.actions[self.pointer] = actions
        self.rewards[self.pointer] = reward
        self.next_obs[self.pointer] = next_obs
        self.dones[self.pointer] = float(done)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> DDPGReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return DDPGReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


def _fanin_uniform(layer: nn.Linear) -> None:
    bound = 1.0 / math.sqrt(layer.weight.shape[1])
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def _final_uniform(layer: nn.Linear) -> None:
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    nn.init.uniform_(layer.bias, -3e-3, 3e-3)


class DDPGActor(nn.Module):
    """Paper actor ``mu(o)`` with bounded continuous output.

    ``obs`` has shape ``(..., obs_dim)`` and the result has shape
    ``(..., action_dim)`` within the elementwise ``[action_low, action_high]``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int] = (400, 300),
        *,
        action_low: float | Tensor = -1.0,
        action_high: float | Tensor = 1.0,
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.input_norm = nn.BatchNorm1d(obs_dim)
        self.fc1 = nn.Linear(obs_dim, first)
        self.first_norm = nn.BatchNorm1d(first)
        self.fc2 = nn.Linear(first, second)
        self.second_norm = nn.BatchNorm1d(second)
        self.output = nn.Linear(second, action_dim)
        _fanin_uniform(self.fc1)
        _fanin_uniform(self.fc2)
        _final_uniform(self.output)

        low = torch.as_tensor(action_low, dtype=torch.float32).expand(action_dim).clone()
        high = torch.as_tensor(action_high, dtype=torch.float32).expand(action_dim).clone()
        if torch.any(high <= low):
            raise ValueError("action_high must be greater than action_low elementwise")
        self.register_buffer("action_midpoint", 0.5 * (high + low))
        self.register_buffer("action_half_range", 0.5 * (high - low))

    def forward(self, obs: Tensor) -> Tensor:
        hidden = F.relu(self.first_norm(self.fc1(self.input_norm(obs))))
        hidden = F.relu(self.second_norm(self.fc2(hidden)))
        return self.action_midpoint + self.action_half_range * torch.tanh(self.output(hidden))


class DDPGCritic(nn.Module):
    """Paper critic ``Q(o, a)`` with action entering at the second hidden layer."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, int] = (400, 300),
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.input_norm = nn.BatchNorm1d(obs_dim)
        self.obs_fc = nn.Linear(obs_dim, first)
        self.obs_hidden_norm = nn.BatchNorm1d(first)
        self.joint_fc = nn.Linear(first + action_dim, second)
        self.output = nn.Linear(second, 1)
        _fanin_uniform(self.obs_fc)
        _fanin_uniform(self.joint_fc)
        _final_uniform(self.output)

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        obs_hidden = F.relu(self.obs_hidden_norm(self.obs_fc(self.input_norm(obs))))
        joint_hidden = F.relu(self.joint_fc(torch.cat([obs_hidden, actions], dim=-1)))
        return self.output(joint_hidden).squeeze(-1)


class _OrnsteinUhlenbeckNoise:
    """Euler discretization of the paper's temporally correlated exploration process."""

    def __init__(self, action_dim: int, theta: float, sigma: float, seed: int) -> None:
        self.theta = theta
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(action_dim, dtype=np.float32)

    def reset(self) -> None:
        self._state.fill(0.0)

    def sample(self) -> np.ndarray:
        innovation = self._rng.standard_normal(self._state.shape).astype(np.float32)
        self._state += self.theta * -self._state + self.sigma * innovation
        return self._state.copy()


class DDPGAgent(nn.Module):
    """One complete continuous DDPG learner for one independently acting agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int] = (400, 300),
        *,
        action_low: float | Tensor = -1.0,
        action_high: float | Tensor = 1.0,
        config: DDPGConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = config or DDPGConfig()
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.actor = DDPGActor(
            obs_dim, action_dim, hidden_dims, action_low=action_low, action_high=action_high,
        )
        self.critic = DDPGCritic(obs_dim, action_dim, hidden_dims)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.target_actor.eval()
        self.target_critic.eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_learning_rate,
            weight_decay=self.config.critic_weight_decay,
        )
        self._noise = _OrnsteinUhlenbeckNoise(
            action_dim, self.config.ou_theta, self.config.ou_sigma, seed,
        )

    def reset_noise(self) -> None:
        """Reset the episode-persistent OU state to its zero mean."""
        self._noise.reset()

    @torch.no_grad()
    def act(self, obs: Tensor, *, explore: bool = True) -> Tensor:
        """Return bounded continuous actions, optionally perturbed by OU noise."""
        was_training = self.actor.training
        self.actor.eval()
        actions = self.actor(obs)
        self.actor.train(was_training)
        if explore:
            noise = torch.as_tensor(self._noise.sample(), device=actions.device, dtype=actions.dtype)
            actions = actions + noise
        low = self.actor.action_midpoint - self.actor.action_half_range
        high = self.actor.action_midpoint + self.actor.action_half_range
        return torch.maximum(torch.minimum(actions, high), low)

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> DDPGUpdate:
        """Apply the paper's one-step critic and deterministic actor updates."""
        if actions.ndim != obs.ndim:
            raise ValueError("DDPG replay actions must be continuous vectors")

        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            target_q = rewards + self.gamma * (1.0 - dones) * self.target_critic(
                next_obs, next_actions,
            )
        critic_loss = F.mse_loss(self.critic(obs, actions), target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        self.critic.requires_grad_(False)
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        self.soft_update()
        return DDPGUpdate(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            target_q=float(target_q.mean()),
        )

    @torch.no_grad()
    def soft_update(self, tau: float | None = None) -> None:
        """Polyak-average actor and critic targets toward their online networks."""
        coefficient = self.tau if tau is None else tau
        soft_update_module(self.target_actor, self.actor, coefficient)
        soft_update_module(self.target_critic, self.critic, coefficient)
        self._update_target_buffers(self.target_actor, self.actor, coefficient)
        self._update_target_buffers(self.target_critic, self.critic, coefficient)

    @staticmethod
    def _update_target_buffers(target: nn.Module, source: nn.Module, tau: float) -> None:
        """Polyak-average BatchNorm statistics alongside target parameters."""
        target_buffers = dict(target.named_buffers())
        for name, source_buffer in source.named_buffers():
            target_buffer = target_buffers[name]
            if source_buffer.is_floating_point():
                target_buffer.mul_(1.0 - tau).add_(source_buffer, alpha=tau)
            else:
                target_buffer.copy_(source_buffer)


__all__ = [
    "DDPGActor",
    "DDPGAgent",
    "DDPGConfig",
    "DDPGCritic",
    "DDPGReplayBatch",
    "DDPGReplayBuffer",
    "DDPGUpdate",
]
