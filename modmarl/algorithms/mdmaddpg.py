"""Memory-Driven MADDPG (MD-MADDPG).

Model: agents sequentially encode private observations, read and update one
episode-persistent shared memory, then choose local actions. Each agent owns a
centralized critic and target actor/critic; ``MDMADDPGLearner`` owns replay,
optimizers, reward normalization, and the 100-step update schedule.
Invariants: memory, observation embedding, and read context are all 200-dimensional;
the context equation is linear; replay stores the exact memory each agent read and
wrote. Tensors use ``(batch, agents, feature)`` ordering.
Interface: ``MDMADDPGConfig`` and ``MDMADDPGLearner`` form the complete discrete-task
API; model pieces remain public for inspection.
Why: the Machine Learning paper explicitly specifies one 512-unit encoder layer,
one 256-unit action-selector layer, and memory width 200. The authors' unreleased
reference code (private Bitbucket repository ``md-maddpg``, revision
``20dffc4e12b5b490ad30defb49887d2599aeece6``; not publicly accessible) instead defaults
to 213, inserts a second 256-unit encoder layer, omits the selector hidden layer, and applies ReLU to
the paper's linear context map; those release discrepancies are corrected. Its
first-batch-only read-gate bug is also corrected by applying every batch row.
Reward normalization guards a zero replay standard deviation instead of emitting
the archived implementation's NaNs.
Validation: the paper's two-agent, 100-step MPE Cooperative Navigation task with
discrete Gumbel actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.replay import MemoryReplayBatch
from ..components import gumbel_policy_sample, soft_update_module


@dataclass(frozen=True)
class MDMADDPGConfig:
    gamma: float = 0.95
    tau: float = 0.01
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    memory_dim: int = 200
    replay_capacity: int = 1_000_000
    batch_size: int = 1024
    update_interval: int = 100
    policy_regularization: float = 1e-3
    memory_regularization: float = 1e-3
    gradient_clip: float = 0.5


@dataclass(frozen=True)
class MDMADDPGUpdate:
    agent_index: int
    critic_loss: float
    actor_loss: float
    target_q_mean: float


class _MDReplayBuffer:
    """Archived replay contract with one reward and termination per agent."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        memory_dim: int,
    ) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
        self.memory_seen = np.zeros(
            (capacity, n_agents, memory_dim), dtype=np.float32,
        )
        self.memory_written = np.zeros_like(self.memory_seen)
        self.size = 0
        self.position = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
        memory_seen: np.ndarray,
        memory_written: np.ndarray,
    ) -> None:
        index = self.position
        self.obs[index] = obs
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.next_obs[index] = next_obs
        self.dones[index] = dones
        self.memory_seen[index] = memory_seen
        self.memory_written[index] = memory_written
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size


class SharedMemory(nn.Module):
    """Episode-persistent memory with the release's fixed random reset value."""

    def __init__(self, memory_dim: int = 200) -> None:
        super().__init__()
        initial = torch.empty(memory_dim)
        nn.init.uniform_(initial, -memory_dim**-0.5, memory_dim**-0.5)
        self.register_buffer("initial_value", initial)

    def reset(self, batch_size: int = 1) -> Tensor:
        return self.initial_value.unsqueeze(0).repeat(batch_size, 1)


class MemoryDrivenActor(nn.Module):
    """Paper Equations 1--8 for one agent's memory-driven policy."""

    def __init__(self, obs_dim: int, action_dim: int, memory_dim: int = 200) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.memory_dim = memory_dim

        self.encoder_hidden = nn.Linear(obs_dim, 512)
        self.encoder_output = nn.Linear(512, memory_dim)
        self.context = nn.Linear(memory_dim, memory_dim, bias=False)
        self.read_gate = nn.Linear(memory_dim * 3, memory_dim, bias=False)
        self.candidate = nn.Linear(memory_dim * 2, memory_dim, bias=False)
        self.input_gate = nn.Linear(memory_dim * 2, memory_dim, bias=False)
        self.forget_gate = nn.Linear(memory_dim * 2, memory_dim, bias=False)
        self.policy_hidden = nn.Linear(memory_dim * 3, 256)
        self.policy_output = nn.Linear(256, action_dim)

    def forward(self, obs: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Return bounded action logits and the memory written by this agent."""
        embedding = self.encoder_output(F.relu(self.encoder_hidden(obs)))

        context = self.context(embedding)
        read_input = torch.cat([embedding, context, memory], dim=-1)
        read_vector = memory * torch.sigmoid(self.read_gate(read_input))

        write_input = torch.cat([embedding, memory], dim=-1)
        candidate = torch.tanh(self.candidate(write_input))
        input_gate = torch.sigmoid(self.input_gate(write_input))
        forget_gate = torch.sigmoid(self.forget_gate(write_input))
        next_memory = input_gate * candidate + forget_gate * memory

        policy_input = torch.cat([embedding, next_memory, read_vector], dim=-1)
        logits = torch.tanh(self.policy_output(F.relu(self.policy_hidden(policy_input))))
        # The released learner treats the memory handed to the next agent as replay
        # state; this agent's gates still learn through ``logits`` above.
        return logits, next_memory.detach()

    def sample(
        self,
        obs: Tensor,
        memory: Tensor,
        *,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample released straight-through Gumbel actions for discrete tasks."""
        logits, next_memory = self(obs, memory)
        one_hot, action_idx, sampled_logits = gumbel_policy_sample(
            logits,
            action_dim=self.action_dim,
            temperature=temperature,
            hard=True,
            deterministic=deterministic,
        )
        return one_hot, action_idx, sampled_logits, next_memory


class MDMADDPGCritic(nn.Module):
    """Centralized paper critic with BatchNorm and 1024-512-256 hidden layers."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        input_dim = n_agents * (obs_dim + action_dim)
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        joint = torch.cat(
            [obs.reshape(obs.shape[0], -1), actions.reshape(actions.shape[0], -1)],
            dim=-1,
        )
        hidden = F.relu(self.fc1(self.input_norm(joint)))
        hidden = F.relu(self.fc2(hidden))
        hidden = F.relu(self.fc3(hidden))
        return self.output(hidden).squeeze(-1)


class MDMADDPGAgent(nn.Module):
    """One memory-driven actor and centralized critic with frozen targets."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        memory_dim: int = 200,
    ) -> None:
        super().__init__()
        self.actor = MemoryDrivenActor(obs_dim, action_dim, memory_dim)
        self.target_actor = MemoryDrivenActor(obs_dim, action_dim, memory_dim)
        self.critic = MDMADDPGCritic(n_agents, obs_dim, action_dim)
        self.target_critic = MDMADDPGCritic(n_agents, obs_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)

    def soft_update(self, tau: float) -> None:
        soft_update_module(self.target_actor, self.actor, tau)
        soft_update_module(self.target_critic, self.critic, tau)


class MDMADDPGLearner(nn.Module):
    """Complete archived learner with paper-correct architecture and memory width."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        *,
        config: MDMADDPGConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.config = config or MDMADDPGConfig()
        self.agents = nn.ModuleList(
            [
                MDMADDPGAgent(
                    n_agents, obs_dim, action_dim, self.config.memory_dim,
                )
                for _ in range(n_agents)
            ]
        )
        self.shared_memory = SharedMemory(self.config.memory_dim)
        self.replay = _MDReplayBuffer(
            self.config.replay_capacity,
            n_agents,
            obs_dim,
            self.config.memory_dim,
        )
        self.actor_optimizers = [
            torch.optim.Adam(
                agent.actor.parameters(), lr=self.config.actor_learning_rate,
            )
            for agent in self.agents
        ]
        self.critic_optimizers = [
            torch.optim.Adam(
                agent.critic.parameters(), lr=self.config.critic_learning_rate,
            )
            for agent in self.agents
        ]
        self.total_steps = 0
        self.register_buffer("memory", self.shared_memory.reset())
        self.eval()

    def reset_memory(self, batch_size: int = 1) -> Tensor:
        self.memory = self.shared_memory.reset(batch_size).to(self.memory.device)
        return self.memory

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Act sequentially and return indices plus memory read/written by each agent."""
        seen, written, action_indices = [], [], []
        memory = self.memory
        for agent_id, agent in enumerate(self.agents):
            seen.append(memory.squeeze(0).clone())
            _, action_idx, _, memory = agent.actor.sample(
                obs[agent_id].unsqueeze(0), memory, deterministic=deterministic,
            )
            written.append(memory.squeeze(0).clone())
            action_indices.append(action_idx)
        self.memory = memory
        return (
            torch.cat(action_indices),
            torch.stack(seen),
            torch.stack(written),
        )

    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
        memory_seen: np.ndarray,
        memory_written: np.ndarray,
    ) -> None:
        self.replay.add(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            memory_seen=memory_seen,
            memory_written=memory_written,
        )
        self.total_steps += 1

    def ready_to_update(self) -> bool:
        return (
            len(self.replay) >= self.config.batch_size
            and self.total_steps % self.config.update_interval == 0
        )

    def update(self) -> list[MDMADDPGUpdate]:
        """Update every agent on an independent normalized replay sample."""
        if not self.ready_to_update():
            return []
        self.train()
        device = next(self.parameters()).device
        updates = []
        for agent_index in range(self.n_agents):
            batch = self._sample_normalized(device)
            updates.append(self._update_agent(batch, agent_index))
            # The archived main loop updates all targets after each agent learner.
            for agent in self.agents:
                agent.soft_update(self.config.tau)
        self.eval()
        return updates

    def _sample_normalized(self, device: torch.device) -> MemoryReplayBatch:
        indices = np.random.choice(
            self.replay.size, size=self.config.batch_size, replace=False,
        )
        filled_rewards = self.replay.rewards[: self.replay.size]
        reward_mean = filled_rewards.mean(axis=0, keepdims=True)
        reward_std = np.maximum(filled_rewards.std(axis=0, keepdims=True), 1e-8)
        return MemoryReplayBatch(
            obs=torch.as_tensor(self.replay.obs[indices], device=device),
            actions=torch.as_tensor(self.replay.actions[indices], device=device),
            rewards=torch.as_tensor(
                (self.replay.rewards[indices] - reward_mean) / reward_std,
                device=device,
            ),
            next_obs=torch.as_tensor(self.replay.next_obs[indices], device=device),
            dones=torch.as_tensor(self.replay.dones[indices], device=device),
            memory_seen=torch.as_tensor(self.replay.memory_seen[indices], device=device),
            memory_written=torch.as_tensor(
                self.replay.memory_written[indices], device=device,
            ),
        )

    def _update_agent(
        self,
        batch: MemoryReplayBatch,
        agent_index: int,
    ) -> MDMADDPGUpdate:
        actions = F.one_hot(batch.actions.long(), self.action_dim).to(torch.float32)

        with torch.no_grad():
            memory = batch.memory_written[:, -1]
            target_actions = []
            for teammate_index, teammate in enumerate(self.agents):
                action, _, _, memory = teammate.target_actor.sample(
                    batch.next_obs[:, teammate_index], memory, deterministic=True,
                )
                target_actions.append(action)
            target_actions_tensor = torch.stack(target_actions, dim=1)

        agent = self.agents[agent_index]
        with torch.no_grad():
            target_q = batch.rewards[:, agent_index] + self.config.gamma * (
                1.0 - batch.dones[:, agent_index]
            ) * agent.target_critic(batch.next_obs, target_actions_tensor)

        critic_loss = F.mse_loss(agent.critic(batch.obs, actions), target_q)
        critic_optimizer = self.critic_optimizers[agent_index]
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.critic.parameters(), self.config.gradient_clip)
        critic_optimizer.step()

        joint_actions = []
        own_logits = None
        own_memory = None
        for teammate_index, teammate in enumerate(self.agents):
            if teammate_index == agent_index:
                action, _, own_logits, own_memory = teammate.actor.sample(
                    batch.obs[:, teammate_index],
                    batch.memory_seen[:, teammate_index],
                )
            else:
                with torch.no_grad():
                    action, _, _, _ = teammate.actor.sample(
                        batch.obs[:, teammate_index],
                        batch.memory_seen[:, teammate_index],
                        deterministic=True,
                    )
            joint_actions.append(action)
        if own_logits is None or own_memory is None:
            raise RuntimeError("agent_index must identify one learner")

        actor_loss = -agent.critic(
            batch.obs, torch.stack(joint_actions, dim=1),
        ).mean()
        actor_loss = actor_loss + self.config.policy_regularization * own_logits.square().mean()
        # This term is gradient-free in the archived release because written memory
        # is detached; retaining it reproduces the reported scalar objective.
        actor_loss = actor_loss + self.config.memory_regularization * own_memory.square().mean()
        actor_optimizer = self.actor_optimizers[agent_index]
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), self.config.gradient_clip)
        actor_optimizer.step()

        return MDMADDPGUpdate(
            agent_index=agent_index,
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            target_q_mean=float(target_q.mean()),
        )


__all__ = [
    "MDMADDPGAgent",
    "MDMADDPGConfig",
    "MDMADDPGCritic",
    "MDMADDPGLearner",
    "MDMADDPGUpdate",
    "MemoryDrivenActor",
    "SharedMemory",
]
