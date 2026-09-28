"""Multi-Actor-Attention-Critic (MAAC).

Model: decentralized categorical actors share a centralized multi-head attention
critic during training. ``MAACLearner`` owns replay, the soft actor-critic update,
optimizers, and the released four-updates-per-100-steps schedule.
Invariants: critic row ``i`` excludes agent ``i`` from attention; shared critic
gradients are divided by the number of agents; target actors and critic receive no
gradients. Tensor contracts use ``(batch, agents, feature)`` ordering.
Interface: ``MAACConfig`` and ``MAACLearner`` are the complete training API;
``MAACActor``, ``MAACAgent``, and ``AttentionCritic`` expose the model pieces.
Why: the ICML paper defines the entropy-adjusted target and counterfactual baseline,
while the official release ``shariqiqbal2810/MAAC@6174a01251251e6778c4ada26bc8d9cd930e3856``
fixes
underspecified architecture and schedule details. Equation 4 keeps entropy inside
the discounted continuation; the release places it outside, which is not reproduced.
Validation: the released three-agent MPE Cooperative Navigation task with
deterministic evaluation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.replay import ReplayBatch
from ..components import soft_update_module


@dataclass(frozen=True)
class MAACConfig:
    """Released experimental settings from ``main.py`` in the pinned source."""

    gamma: float = 0.99
    tau: float = 0.001
    entropy_temperature: float = 0.01
    policy_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    critic_weight_decay: float = 1e-3
    policy_regularization: float = 1e-3
    batch_size: int = 1024
    replay_capacity: int = 1_000_000
    update_interval: int = 100
    updates_per_interval: int = 4
    policy_gradient_clip: float = 0.5


@dataclass(frozen=True)
class MAACUpdate:
    critic_loss: float
    actor_loss: float
    target_q_mean: float


@dataclass(frozen=True)
class MAACCriticOutput:
    q_taken: Tensor
    all_q: Tensor
    attention: list[list[Tensor]] | None
    attention_regularization: Tensor


class MAACActor(nn.Module):
    """Released three-layer categorical policy with input BatchNorm."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.input_norm = nn.BatchNorm1d(obs_dim, affine=False)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        hidden = F.leaky_relu(self.fc1(self.input_norm(obs)))
        hidden = F.leaky_relu(self.fc2(hidden))
        return self.fc3(hidden)

    def sample(
        self,
        obs: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample categorical actions and return the released policy statistics."""
        logits = self(obs)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        if deterministic:
            action_idx = logits.argmax(dim=-1)
        else:
            action_idx = torch.multinomial(probs, 1).squeeze(-1)
        one_hot = F.one_hot(action_idx, self.action_dim).to(dtype=logits.dtype)
        chosen_log_prob = log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return one_hot, action_idx, logits, probs, log_probs, chosen_log_prob, entropy


class MAACAgent(nn.Module):
    """One decentralized actor and its frozen target copy."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = MAACActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.requires_grad_(False)

    def soft_update(self, tau: float) -> None:
        soft_update_module(self.target_actor, self.actor, tau)


class AttentionCritic(nn.Module):
    """Centralized attention critic producing each agent's action-value vector."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        attend_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % attend_heads != 0:
            raise ValueError("hidden_dim must be divisible by attend_heads")
        self.n_agents = n_agents
        self.attend_heads = attend_heads

        attend_dim = hidden_dim // attend_heads
        self.state_action_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(obs_dim + action_dim, affine=False),
                    nn.Linear(obs_dim + action_dim, hidden_dim),
                    nn.LeakyReLU(),
                )
                for _ in range(n_agents)
            ]
        )
        self.state_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(obs_dim, affine=False),
                    nn.Linear(obs_dim, hidden_dim),
                    nn.LeakyReLU(),
                )
                for _ in range(n_agents)
            ]
        )
        self.agent_q_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(n_agents)
            ]
        )
        self.key_extractors = nn.ModuleList(
            [nn.Linear(hidden_dim, attend_dim, bias=False) for _ in range(attend_heads)]
        )
        self.selector_extractors = nn.ModuleList(
            [nn.Linear(hidden_dim, attend_dim, bias=False) for _ in range(attend_heads)]
        )
        self.value_extractors = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU())
                for _ in range(attend_heads)
            ]
        )

    def shared_parameters(self):
        return chain(
            self.state_action_encoders.parameters(),
            self.key_extractors.parameters(),
            self.selector_extractors.parameters(),
            self.value_extractors.parameters(),
        )

    def scale_shared_grads(self) -> None:
        for parameter in self.shared_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(1.0 / self.n_agents)

    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
        *,
        return_attention: bool = False,
    ) -> MAACCriticOutput:
        """Evaluate joint transitions.

        ``obs`` has shape ``(B, N, O)`` and one-hot ``actions`` has shape
        ``(B, N, A)``. Returned Q tensors have shapes ``(B, N)`` and
        ``(B, N, A)``.
        """
        sa_inputs = torch.cat([obs, actions], dim=-1)
        sa_encodings = [
            encoder(sa_inputs[:, agent_id])
            for agent_id, encoder in enumerate(self.state_action_encoders)
        ]
        state_encodings = [
            encoder(obs[:, agent_id])
            for agent_id, encoder in enumerate(self.state_encoders)
        ]

        per_agent_values: list[list[Tensor]] = [[] for _ in range(self.n_agents)]
        per_agent_attention: list[list[Tensor]] = [[] for _ in range(self.n_agents)]
        attention_regularization = obs.new_zeros(())

        for head_id in range(self.attend_heads):
            keys = [self.key_extractors[head_id](encoding) for encoding in sa_encodings]
            values = [self.value_extractors[head_id](encoding) for encoding in sa_encodings]
            selectors = [self.selector_extractors[head_id](encoding) for encoding in state_encodings]
            scale = math.sqrt(keys[0].shape[-1])

            for agent_id in range(self.n_agents):
                other_keys = [key for other_id, key in enumerate(keys) if other_id != agent_id]
                other_values = [value for other_id, value in enumerate(values) if other_id != agent_id]
                if not other_keys:
                    per_agent_values[agent_id].append(torch.zeros_like(values[agent_id]))
                    if return_attention:
                        per_agent_attention[agent_id].append(obs.new_zeros((obs.shape[0], 0)))
                    continue

                selector = selectors[agent_id].unsqueeze(1)
                key_tensor = torch.stack(other_keys, dim=1)
                raw_logits = torch.matmul(selector, key_tensor.transpose(1, 2))
                logits = raw_logits / scale
                weights = torch.softmax(logits, dim=-1)
                value_tensor = torch.stack(other_values, dim=1)
                per_agent_values[agent_id].append(
                    (weights.transpose(1, 2) * value_tensor).sum(dim=1)
                )
                attention_regularization = (
                    attention_regularization + 1e-3 * raw_logits.square().mean()
                )
                if return_attention:
                    per_agent_attention[agent_id].append(weights.squeeze(1))

        all_q_values = []
        chosen_q_values = []
        for agent_id in range(self.n_agents):
            context = torch.cat(per_agent_values[agent_id], dim=-1)
            q_values = self.agent_q_heads[agent_id](
                torch.cat([state_encodings[agent_id], context], dim=-1)
            )
            all_q_values.append(q_values)
            chosen_idx = actions[:, agent_id].argmax(dim=-1, keepdim=True)
            chosen_q_values.append(q_values.gather(1, chosen_idx).squeeze(-1))

        return MAACCriticOutput(
            q_taken=torch.stack(chosen_q_values, dim=1),
            all_q=torch.stack(all_q_values, dim=1),
            attention=per_agent_attention if return_attention else None,
            attention_regularization=attention_regularization,
        )


class _MAACReplayBuffer:
    """Joint replay retaining the per-agent rewards required by MAAC."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
        self.size = 0
        self.pointer = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
    ) -> None:
        self.obs[self.pointer] = obs
        self.actions[self.pointer] = actions
        self.rewards[self.pointer] = rewards
        self.next_obs[self.pointer] = next_obs
        self.dones[self.pointer] = dones
        self.pointer = (self.pointer + 1) % self.capacity
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


class MAACLearner(nn.Module):
    """Complete paper/release learner, including replay and update cadence."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 128,
        attend_heads: int = 4,
        config: MAACConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.config = config or MAACConfig()
        self.agents = nn.ModuleList(
            [MAACAgent(obs_dim, action_dim, hidden_dim) for _ in range(n_agents)]
        )
        self.critic = AttentionCritic(
            n_agents, obs_dim, action_dim, hidden_dim, attend_heads,
        )
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.actor_optimizers = [
            torch.optim.Adam(agent.actor.parameters(), lr=self.config.policy_learning_rate)
            for agent in self.agents
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_learning_rate,
            weight_decay=self.config.critic_weight_decay,
        )
        self.replay = _MAACReplayBuffer(
            self.config.replay_capacity, n_agents, obs_dim,
        )
        self.total_steps = 0
        self.eval()

    @torch.no_grad()
    def act(self, obs: Tensor, *, deterministic: bool = False) -> Tensor:
        """Return environment action indices for local observations ``(N, O)``."""
        return torch.cat(
            [
                agent.actor.sample(
                    obs[agent_id].unsqueeze(0), deterministic=deterministic,
                )[1]
                for agent_id, agent in enumerate(self.agents)
            ]
        )

    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
    ) -> None:
        self.replay.add(obs, actions, rewards, next_obs, dones)
        self.total_steps += 1

    def ready_to_update(self) -> bool:
        return (
            len(self.replay) >= self.config.batch_size
            and self.total_steps % self.config.update_interval == 0
        )

    def update(self) -> list[MAACUpdate]:
        """Apply exactly four learner updates at each released update boundary."""
        if not self.ready_to_update():
            return []
        self.train()
        device = next(self.parameters()).device
        updates = [
            self._update_batch(self.replay.sample(self.config.batch_size, device))
            for _ in range(self.config.updates_per_interval)
        ]
        self.eval()
        return updates

    def _update_batch(self, batch: ReplayBatch) -> MAACUpdate:
        actions = F.one_hot(batch.actions.long(), self.action_dim).to(torch.float32)

        with torch.no_grad():
            next_samples = [
                agent.target_actor.sample(batch.next_obs[:, agent_id])
                for agent_id, agent in enumerate(self.agents)
            ]
            next_actions = torch.stack([sample[0] for sample in next_samples], dim=1)
            next_log_prob = torch.stack([sample[5] for sample in next_samples], dim=1)
            next_q = self.target_critic(batch.next_obs, next_actions).q_taken
            target_q = batch.rewards + self.config.gamma * (1.0 - batch.dones) * (
                next_q - self.config.entropy_temperature * next_log_prob
            )

        critic_output = self.critic(batch.obs, actions)
        critic_loss = sum(
            F.mse_loss(critic_output.q_taken[:, agent_id], target_q[:, agent_id])
            for agent_id in range(self.n_agents)
        ) + critic_output.attention_regularization
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic.scale_shared_grads()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10 * self.n_agents)
        self.critic_optimizer.step()

        samples = [
            agent.actor.sample(batch.obs[:, agent_id])
            for agent_id, agent in enumerate(self.agents)
        ]
        with torch.no_grad():
            joint_actions = torch.stack([sample[0] for sample in samples], dim=1)
            policy_q = self.critic(batch.obs, joint_actions)

        actor_losses = []
        for agent_id, (agent, optimizer, sample) in enumerate(
            zip(self.agents, self.actor_optimizers, samples)
        ):
            _, _, logits, probs, _, log_prob, _ = sample
            baseline = (policy_q.all_q[:, agent_id] * probs).sum(dim=-1)
            advantage = policy_q.q_taken[:, agent_id] - baseline
            actor_loss = (
                log_prob
                * (self.config.entropy_temperature * log_prob - advantage).detach()
            ).mean()
            actor_loss = actor_loss + self.config.policy_regularization * logits.square().mean()
            optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                agent.actor.parameters(), self.config.policy_gradient_clip,
            )
            optimizer.step()
            actor_losses.append(actor_loss.detach())

        soft_update_module(self.target_critic, self.critic, self.config.tau)
        for agent in self.agents:
            agent.soft_update(self.config.tau)

        return MAACUpdate(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(torch.stack(actor_losses).mean()),
            target_q_mean=float(target_q.mean()),
        )


__all__ = [
    "AttentionCritic",
    "MAACActor",
    "MAACAgent",
    "MAACConfig",
    "MAACCriticOutput",
    "MAACLearner",
    "MAACUpdate",
]
