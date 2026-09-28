"""Multi-Agent Deep Deterministic Policy Gradient (MADDPG).

Model: each agent owns a local actor and a centralized action-value critic.  The
critic observes every agent's observation and action during training; execution
uses only the corresponding actor's local observation.  For discrete MPE tasks,
actions are differentiable Gumbel-Softmax samples, exactly as in the released
OpenAI implementation.

Invariants: agents use independent parameters and rewards, replay retains the
full sampled action vectors, actor updates replace only the learning agent's
replayed action, and target networks move by Polyak averaging after each update.
Interface: ``MADDPGAgent.act`` selects one local action and ``MADDPGAgent.update``
performs the complete paper Eq. 5--6 update using the other agents as context.

Paper: Lowe et al., *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
Environments*, NeurIPS 2017.  This module follows openai/maddpg commit 3ceefa0:
two 64-unit ReLU layers, soft categorical sampling, same-batch centralized
critics, policy-logit regularization, 0.5 gradient clipping, and tau=0.01.
The storage-only adaptation is one joint replay array instead of synchronized
per-agent arrays; every learner still samples an independent aligned joint
batch and uses its own reward and done flag.  Targets start as online-network
copies, the standard DDPG initialization implied by the paper, rather than as
unrelated random networks before the release's first soft update.
Policy ensembles and opponent-policy estimation are paper experiments layered
on MADDPG, not part of the released core trainer, and are therefore not silently
folded into the algorithm defined here.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..components import soft_update_module
from ..components.critics import CentralizedMLPCritic
from ..components.policies import DiscreteMLPActor


@dataclass(frozen=True)
class MADDPGConfig:
    """Reference hyperparameters for the released discrete-MPE trainer."""

    learning_rate: float = 1e-2
    gamma: float = 0.95
    tau: float = 0.01
    policy_regularization: float = 1e-3
    gradient_clip: float = 0.5
    batch_size: int = 1024
    replay_capacity: int = 1_000_000
    max_episode_len: int = 25
    update_interval: int = 100
    minimum_replay_size: int | None = None


@dataclass(frozen=True)
class MADDPGUpdate:
    """Scalar diagnostics from one agent's actor-critic update."""

    critic_loss: float
    actor_loss: float
    target_q_mean: float
    reward_mean: float


@dataclass(frozen=True)
class MADDPGReplayBatch:
    """Joint transition batch; actions are sampled vectors, not integer labels."""

    obs: Tensor
    actions: Tensor
    rewards: Tensor
    next_obs: Tensor
    dones: Tensor


class MADDPGReplayBuffer:
    """Algorithm-specific replay preserving per-agent rewards and soft actions."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = dones
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> MADDPGReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return MADDPGReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


class MADDPGActor(DiscreteMLPActor):
    """Local two-layer actor producing a differentiable categorical action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        _initialize_like_tensorflow(self)


class MADDPGCritic(CentralizedMLPCritic):
    """Centralized critic over the complete joint observation and action."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        _initialize_like_tensorflow(self)


class MADDPGAgent(nn.Module):
    """One full MADDPG learner; construct one instance per environment agent."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        config: MADDPGConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.config = config or MADDPGConfig()
        self.actor = MADDPGActor(obs_dim, action_dim, hidden_dim)
        self.critic = MADDPGCritic(n_agents, obs_dim, action_dim, hidden_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        for target in (self.target_actor, self.target_critic):
            target.requires_grad_(False)

    def act(self, obs: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Return sampled action vectors and their environment action indices."""
        action, index, _ = self.actor.sample(obs, hard=False, deterministic=deterministic)
        return action, index

    def update(
        self,
        agents: Sequence[MADDPGAgent],
        agent_index: int,
        batch: MADDPGReplayBatch,
    ) -> MADDPGUpdate:
        """Apply the released centralized-critic and local-actor update (Eq. 5--6)."""
        if len(agents) != self.n_agents:
            raise ValueError("agents must contain exactly n_agents learners")

        # Eq. 6: target policies jointly act at the next observations.
        with torch.no_grad():
            target_actions = torch.stack(
                [agent.target_actor.sample(batch.next_obs[:, i], hard=False)[0] for i, agent in enumerate(agents)],
                dim=1,
            )
            reward = batch.rewards[:, agent_index]
            done = batch.dones[:, agent_index]
            target_q = reward + self.config.gamma * (1.0 - done) * self.target_critic(
                batch.next_obs, target_actions
            )

        critic_loss = F.mse_loss(self.critic(batch.obs, batch.actions), target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()

        # Eq. 5: replace only this actor's replayed action; teammate actions stay fixed.
        own_action, _, logits = self.actor.sample(batch.obs[:, agent_index], hard=False)
        joint_actions = batch.actions.clone()
        joint_actions[:, agent_index] = own_action
        actor_loss = -self.critic(batch.obs, joint_actions).mean()
        actor_loss = actor_loss + self.config.policy_regularization * logits.square().mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()

        self.soft_update()
        return MADDPGUpdate(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            target_q_mean=float(target_q.mean()),
            reward_mean=float(reward.mean()),
        )

    def soft_update(self, tau: float | None = None) -> None:
        """Polyak-update actor and critic targets using reference tau=0.01 by default."""
        coefficient = self.config.tau if tau is None else tau
        soft_update_module(self.target_actor, self.actor, coefficient)
        soft_update_module(self.target_critic, self.critic, coefficient)


class MADDPGLearner(nn.Module):
    """Complete multi-agent trainer, including replay and the released schedule."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        config: MADDPGConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MADDPGConfig()
        self.agents = nn.ModuleList(
            [MADDPGAgent(n_agents, obs_dim, action_dim, hidden_dim, self.config) for _ in range(n_agents)]
        )
        self.replay = MADDPGReplayBuffer(
            self.config.replay_capacity, n_agents, obs_dim, action_dim
        )
        self.total_steps = 0

    @torch.no_grad()
    def act(self, obs: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Act jointly while every actor observes only its own observation."""
        samples = [agent.act(obs[i].unsqueeze(0), deterministic=deterministic) for i, agent in enumerate(self.agents)]
        actions = torch.cat([sample[0] for sample in samples], dim=0)
        indices = torch.cat([sample[1] for sample in samples], dim=0)
        return actions, indices

    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
    ) -> None:
        """Append one joint transition and advance the reference global clock."""
        self.replay.add(obs, actions, rewards, next_obs, dones)
        self.total_steps += 1

    def ready_to_update(self) -> bool:
        """Match OpenAI's batch*horizon warmup and one update per 100 steps."""
        minimum_replay = self.config.minimum_replay_size
        if minimum_replay is None:
            minimum_replay = self.config.batch_size * self.config.max_episode_len
        return len(self.replay) >= minimum_replay and self.total_steps % self.config.update_interval == 0

    def update(self) -> list[MADDPGUpdate]:
        """Update each agent from its own sampled joint batch, as in the release."""
        if not self.ready_to_update():
            return []
        device = next(self.parameters()).device
        return [
            agent.update(self.agents, i, self.replay.sample(self.config.batch_size, device))
            for i, agent in enumerate(self.agents)
        ]


def _initialize_like_tensorflow(module: nn.Module) -> None:
    """Match TensorFlow contrib fully_connected's Xavier weights and zero biases."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


__all__ = [
    "MADDPGActor",
    "MADDPGAgent",
    "MADDPGConfig",
    "MADDPGCritic",
    "MADDPGLearner",
    "MADDPGReplayBatch",
    "MADDPGReplayBuffer",
    "MADDPGUpdate",
]
