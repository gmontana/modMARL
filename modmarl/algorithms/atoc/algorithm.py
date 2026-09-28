"""Attentional Communication (ATOC) for continuous cooperative control.

Model: one parameter-shared DDPG actor produces a 128-dimensional thought for each
agent.  A separately supervised attention unit chooses initiators every ``T`` steps;
nearby collaborators enter persistent groups whose thoughts are integrated by a
bidirectional LSTM before the remaining actor layers produce continuous actions.
Invariants: groups contain their initiator, use only eligible neighbours, persist for
the configured communication period, and overlapping groups execute sequentially so a
shared agent carries its first integrated thought into the next group.  The critic is
local and parameter-shared, matching the paper rather than silently substituting a
MADDPG centralized critic.  An agent outside every active group carries its local thought
into ActorNet II; communication replaces that second input only for participating agents.
Initiator rows and their members use ascending agent-index order, making the paper's
otherwise task-defined BiLSTM ordering deterministic.
Interface: ``ATOCGroupScheduler`` owns execution-time groups, ``ATOCPolicy`` owns the
actor/attention/channel tensor flow, and ``ATOCLearner`` owns replay, DDPG updates, OU
exploration, and the episode-normalized attention-classifier update.
Why: Jiang and Lu, *Learning Attentional Communication for Multi-Agent Cooperation*,
NeurIPS 2018 (arXiv:1805.07733), specifies no author code release.  The paper fixes the
thought size, four actor layers, critic sizes, BiLSTM channel, optimizer, and schedule,
but not the other actor widths, attention/channel widths, collaborator cap,
normal-initialization scale, or exact BatchNorm placement; the explicit defaults below
use ``(256, 128, 128, 64)``, 64-unit attention and channel layers, at most three
collaborators, variance-scaled normal weights, and BatchNorm on the critic's first hidden
layer.  No unverifiable third-party release is treated as authoritative.  Validation uses the
repository's bounded signed-action noisy-navigation task, where the gifted agent's true
landmark observations make communication useful, not the paper-scale 50-agent experiment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...components import soft_update_module

PAPER_SOURCE = "paper:neurips-2018-atoc;author-code:none"


@dataclass(frozen=True)
class ATOCConfig:
    """Paper settings, with explicit widths for the paper's unspecified layers."""

    actor_hidden_dims: tuple[int, int, int, int] = (256, 128, 128, 64)
    critic_hidden_dims: tuple[int, int] = (512, 256)
    attention_hidden_dim: int = 64
    channel_hidden_dim: int = 64
    communication_period: int = 15
    max_collaborators: int = 3
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    attention_learning_rate: float = 1e-3
    gamma: float = 0.96
    tau: float = 1e-3
    replay_capacity: int = 100_000
    batch_size: int = 2_560
    warmup_episodes: int = 30
    ou_theta: float = 0.15
    ou_sigma: float = 0.2

    @property
    def thought_dim(self) -> int:
        return self.actor_hidden_dims[1]


@dataclass(frozen=True)
class ATOCReplayBatch:
    """Joint transition batch with paper communication matrices ``C``."""

    obs: Tensor
    actions: Tensor
    rewards: Tensor
    next_obs: Tensor
    dones: Tensor
    groups: Tensor


@dataclass(frozen=True)
class ATOCOutput:
    """One joint execution step."""

    actions: Tensor
    thoughts: Tensor
    integrated_thoughts: Tensor
    attention_probabilities: Tensor
    groups: Tensor


@dataclass(frozen=True)
class ATOCUpdate:
    critic_loss: float
    actor_loss: float
    target_q_mean: float


class ATOCReplayBuffer:
    """Fixed transition replay retaining the active group matrix for each action."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
        self.groups = np.zeros((capacity, n_agents, n_agents), dtype=bool)
        self.size = 0
        self.position = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
        groups: np.ndarray,
    ) -> None:
        index = self.position
        self.obs[index] = obs
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.next_obs[index] = next_obs
        self.dones[index] = dones
        self.groups[index] = groups
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ATOCReplayBatch:
        if self.size < batch_size:
            raise ValueError("replay does not contain one complete minibatch")
        indices = np.random.randint(0, self.size, size=batch_size)
        return ATOCReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
            groups=torch.as_tensor(self.groups[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"ATOCReplayBuffer(size={self.size}, capacity={self.capacity})"


class ATOCGroupScheduler:
    """Form and persist the paper's proximity-prioritized communication groups."""

    def __init__(self, n_agents: int, period: int = 15, max_collaborators: int = 3) -> None:
        if period < 1:
            raise ValueError("period must be positive")
        if max_collaborators < 1:
            raise ValueError("max_collaborators must be positive")
        self.n_agents = n_agents
        self.period = period
        self.max_collaborators = max_collaborators
        self._groups = np.zeros((n_agents, n_agents), dtype=bool)
        self._remaining = 0

    def reset(self) -> None:
        self._groups.fill(False)
        self._remaining = 0

    def step(
        self,
        initiators: np.ndarray,
        distances: np.ndarray,
        eligible: np.ndarray,
    ) -> np.ndarray:
        """Return ``(initiator, member)`` groups, re-forming only every ``period`` steps."""
        initiators = np.asarray(initiators, dtype=bool)
        distances = np.asarray(distances, dtype=np.float32)
        eligible = np.asarray(eligible, dtype=bool)
        if initiators.shape != (self.n_agents,):
            raise ValueError("initiators must have shape (n_agents,)")
        expected = (self.n_agents, self.n_agents)
        if distances.shape != expected or eligible.shape != expected:
            raise ValueError("distances and eligible must have shape (n_agents, n_agents)")
        if self._remaining:
            self._remaining -= 1
            return self._groups.copy()

        groups = np.zeros_like(self._groups)
        already_selected = np.zeros(self.n_agents, dtype=bool)
        for initiator in np.flatnonzero(initiators):
            candidates = np.flatnonzero(eligible[initiator] & (np.arange(self.n_agents) != initiator))
            if not candidates.size:
                continue
            categories = np.where(
                initiators[candidates],
                2,
                np.where(already_selected[candidates], 1, 0),
            )
            order = np.lexsort((distances[initiator, candidates], categories))
            collaborators = candidates[order[: self.max_collaborators]]
            groups[initiator, initiator] = True
            groups[initiator, collaborators] = True
            already_selected[collaborators] = True
        self._groups = groups
        self._remaining = self.period - 1
        return groups.copy()

    def __repr__(self) -> str:
        active = int(self._groups.any(axis=1).sum())
        return (
            f"ATOCGroupScheduler(n_agents={self.n_agents}, period={self.period}, "
            f"active_groups={active}, remaining={self._remaining})"
        )


class ATOCActor(nn.Module):
    """Four-hidden-layer actor split at the paper's 128-unit thought layer."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int, int, int] = (256, 128, 128, 64),
        *,
        action_low: float | Tensor = -1.0,
        action_high: float | Tensor = 1.0,
    ) -> None:
        super().__init__()
        first, thought, third, fourth = hidden_dims
        self.first = nn.Linear(obs_dim, first)
        self.thought_layer = nn.Linear(first, thought)
        self.third = nn.Linear(2 * thought, third)
        self.fourth = nn.Linear(third, fourth)
        self.action_head = nn.Linear(fourth, action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32).expand(action_dim).clone()
        high = torch.as_tensor(action_high, dtype=torch.float32).expand(action_dim).clone()
        if torch.any(high <= low):
            raise ValueError("action_high must exceed action_low")
        self.register_buffer("action_midpoint", (low + high) / 2)
        self.register_buffer("action_half_range", (high - low) / 2)
        _normal_initialization(self)

    def encode(self, obs: Tensor) -> Tensor:
        """Map ``(..., obs_dim)`` observations to paper thoughts."""
        shape = obs.shape[:-1]
        flat = obs.reshape(-1, obs.shape[-1])
        hidden = F.relu(self.first(flat))
        return F.relu(self.thought_layer(hidden)).reshape(*shape, -1)

    def act_from_thought(self, thought: Tensor, integrated: Tensor) -> Tensor:
        hidden = F.relu(self.third(torch.cat([thought, integrated], dim=-1)))
        hidden = F.relu(self.fourth(hidden))
        squashed = torch.tanh(self.action_head(hidden))
        return self.action_midpoint + self.action_half_range * squashed


class ATOCAttentionUnit(nn.Module):
    """Two-layer communication-probability classifier over an agent thought."""

    def __init__(self, thought_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(thought_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        _normal_initialization(self)

    def forward(self, thought: Tensor) -> Tensor:
        return torch.sigmoid(self.net(thought).squeeze(-1))


class ATOCCommunicationChannel(nn.Module):
    """Sequential overlapping-group BiLSTM communication channel."""

    def __init__(self, thought_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.channel = nn.LSTM(
            thought_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        output_dim = 2 * hidden_dim
        self.output = nn.Identity() if output_dim == thought_dim else nn.Linear(output_dim, thought_dim)
        _normal_initialization(self)

    def forward(self, thoughts: Tensor, groups: Tensor) -> Tensor:
        """Integrate ``(B,N,H)`` thoughts using ``(B,N,N)`` initiator rows."""
        if groups.shape != thoughts.shape[:2] + (thoughts.shape[1],):
            raise ValueError("groups must have shape (batch, n_agents, n_agents)")
        carried = thoughts
        integrated = torch.zeros_like(thoughts)
        agent_order = torch.arange(thoughts.shape[1], device=thoughts.device)
        for initiator in range(groups.shape[1]):
            membership = groups[:, initiator].to(dtype=torch.bool)
            lengths = membership.sum(dim=-1)
            active = torch.nonzero(lengths >= 2, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue
            active_membership = membership.index_select(0, active)
            # Selected agent indices lead, in ascending order; padding follows.
            sort_keys = (~active_membership).to(torch.long) * thoughts.shape[1] + agent_order
            member_order = sort_keys.argsort(dim=-1)
            active_carried = carried.index_select(0, active)
            sequence = active_carried.gather(
                1,
                member_order.unsqueeze(-1).expand(-1, -1, thoughts.shape[-1]),
            )
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence,
                lengths.index_select(0, active).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.channel(packed)
            padded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=thoughts.shape[1],
            )
            padded = self.output(padded)
            reordered = torch.zeros_like(active_carried).scatter(
                1,
                member_order.unsqueeze(-1).expand_as(padded),
                padded,
            )
            next_carried = torch.where(
                active_membership.unsqueeze(-1), reordered, active_carried,
            )
            next_integrated = torch.where(
                active_membership.unsqueeze(-1),
                reordered,
                integrated.index_select(0, active),
            )
            carried = carried.index_copy(0, active, next_carried)
            integrated = integrated.index_copy(0, active, next_integrated)
        return integrated


class ATOCPolicy(nn.Module):
    """Shared actor, attention classifier, and communication channel."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: ATOCConfig | None = None,
        *,
        action_low: float | Tensor = -1.0,
        action_high: float | Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.config = config or ATOCConfig()
        self.actor = ATOCActor(
            obs_dim,
            action_dim,
            self.config.actor_hidden_dims,
            action_low=action_low,
            action_high=action_high,
        )
        self.attention = ATOCAttentionUnit(
            self.config.thought_dim,
            self.config.attention_hidden_dim,
        )
        self.channel = ATOCCommunicationChannel(
            self.config.thought_dim,
            self.config.channel_hidden_dim,
        )

    def forward(self, obs: Tensor, groups: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return actions, local thoughts, and integrated thoughts for a joint batch."""
        thoughts = self.actor.encode(obs)
        groups = groups.to(dtype=torch.bool)
        communicated = self.channel(thoughts, groups)
        participates = groups.any(dim=1).unsqueeze(-1)
        integrated = torch.where(participates, communicated, thoughts)
        return self.actor.act_from_thought(thoughts, integrated), thoughts, integrated

    def independent_actions(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        thoughts = self.actor.encode(obs)
        return self.actor.act_from_thought(thoughts, thoughts), thoughts


class ATOCCritic(nn.Module):
    """Paper local action-value function, shared across homogeneous agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int] = (512, 256),
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.obs_layer = nn.Linear(obs_dim, first)
        self.hidden_norm = nn.BatchNorm1d(first)
        self.joint_layer = nn.Linear(first + action_dim, second)
        self.value = nn.Linear(second, 1)
        _normal_initialization(self)

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])
        hidden = F.relu(_safe_batch_norm(self.hidden_norm, self.obs_layer(flat_obs)))
        hidden = F.relu(self.joint_layer(torch.cat([hidden, flat_actions], dim=-1)))
        return self.value(hidden).reshape(*shape)


class ATOCLearner(nn.Module):
    """Complete shared ATOC learner with replay and separate attention supervision."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        config: ATOCConfig | None = None,
        *,
        action_low: float | Tensor = -1.0,
        action_high: float | Tensor = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.config = config or ATOCConfig()
        self.policy = ATOCPolicy(
            obs_dim,
            action_dim,
            self.config,
            action_low=action_low,
            action_high=action_high,
        )
        self.critic = ATOCCritic(obs_dim, action_dim, self.config.critic_hidden_dims)
        self.target_policy = copy.deepcopy(self.policy).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            [*self.policy.actor.parameters(), *self.policy.channel.parameters()],
            lr=self.config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate,
        )
        self.attention_optimizer = torch.optim.Adam(
            self.policy.attention.parameters(), lr=self.config.attention_learning_rate,
        )
        self.replay = ATOCReplayBuffer(
            self.config.replay_capacity,
            n_agents,
            obs_dim,
            action_dim,
        )
        self._noise_rng = np.random.default_rng(seed)
        self._noise_state = np.zeros((n_agents, action_dim), dtype=np.float32)

    def reset_noise(self) -> None:
        self._noise_state.fill(0.0)

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        scheduler: ATOCGroupScheduler,
        distances: np.ndarray,
        eligible: np.ndarray,
        *,
        deterministic: bool = False,
        explore: bool = True,
    ) -> ATOCOutput:
        """Select initiators, update persistent groups, and return joint actions."""
        was_training = self.policy.training
        self.policy.eval()
        thoughts = self.policy.actor.encode(obs.unsqueeze(0))
        probabilities = self.policy.attention(thoughts).squeeze(0)
        initiators = probabilities >= 0.5
        groups_np = scheduler.step(
            initiators.cpu().numpy(), distances, eligible,
        )
        groups = torch.as_tensor(groups_np, device=obs.device).unsqueeze(0)
        actions, thoughts, integrated = self.policy(obs.unsqueeze(0), groups)
        self.policy.train(was_training)
        actions = actions.squeeze(0)
        if explore and not deterministic:
            innovation = self._noise_rng.standard_normal(self._noise_state.shape).astype(np.float32)
            self._noise_state += (
                -self.config.ou_theta * self._noise_state
                + self.config.ou_sigma * innovation
            )
            actions = actions + torch.as_tensor(
                self._noise_state, device=actions.device, dtype=actions.dtype,
            )
        low = self.policy.actor.action_midpoint - self.policy.actor.action_half_range
        high = self.policy.actor.action_midpoint + self.policy.actor.action_half_range
        actions = torch.maximum(torch.minimum(actions, high), low)
        return ATOCOutput(
            actions=actions,
            thoughts=thoughts.squeeze(0),
            integrated_thoughts=integrated.squeeze(0),
            attention_probabilities=probabilities,
            groups=groups.squeeze(0),
        )

    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
        groups: np.ndarray,
    ) -> None:
        self.replay.add(obs, actions, rewards, next_obs, dones, groups)

    def update(self, batch: ATOCReplayBatch | None = None) -> ATOCUpdate | None:
        """Apply the paper's shared DDPG critic, actor, channel, and target updates."""
        if batch is None:
            if len(self.replay) < self.config.batch_size:
                return None
            batch = self.replay.sample(self.config.batch_size, next(self.parameters()).device)

        self.target_policy.eval()
        self.target_critic.eval()
        with torch.no_grad():
            next_actions, _, _ = self.target_policy(batch.next_obs, batch.groups)
            target_q = batch.rewards + self.config.gamma * (1.0 - batch.dones) * (
                self.target_critic(batch.next_obs, next_actions)
            )
        critic_q = self.critic(batch.obs, batch.actions)
        critic_loss = F.mse_loss(critic_q, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        policy_actions, _, _ = self.policy(batch.obs, batch.groups)
        actor_loss = -self.critic(batch.obs, policy_actions).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        self.soft_update()
        return ATOCUpdate(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            target_q_mean=float(target_q.mean()),
        )

    def update_attention_episode(
        self,
        obs: Tensor,
        coordinated_actions: Tensor,
        groups: Tensor,
    ) -> float | None:
        """Train attention from episode-min-max-normalized mean group ΔQ labels."""
        policy_was_training = self.policy.training
        critic_was_training = self.critic.training
        self.policy.eval()
        self.critic.eval()
        with torch.no_grad():
            independent_actions, thoughts = self.policy.independent_actions(obs)
            coordinated_q = self.critic(obs, coordinated_actions)
            independent_q = self.critic(obs, independent_actions)
            improvement = coordinated_q - independent_q
            examples: list[Tensor] = []
            deltas: list[Tensor] = []
            for time in range(obs.shape[0]):
                for initiator in range(self.n_agents):
                    members = groups[time, initiator].to(dtype=torch.bool)
                    if members.sum() < 2:
                        continue
                    examples.append(thoughts[time, initiator])
                    deltas.append(improvement[time, members].mean())
        if not examples:
            self.policy.train(policy_was_training)
            self.critic.train(critic_was_training)
            return None
        thought_batch = torch.stack(examples)
        delta_batch = torch.stack(deltas)
        span = delta_batch.max() - delta_batch.min()
        labels = (
            (delta_batch - delta_batch.min()) / span
            if float(span) > torch.finfo(span.dtype).eps
            else torch.zeros_like(delta_batch)
        )
        self.policy.train(policy_was_training)
        self.critic.train(critic_was_training)
        predictions = self.policy.attention(thought_batch.detach())
        loss = F.binary_cross_entropy(predictions, labels.detach())
        self.attention_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.attention_optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def soft_update(self) -> None:
        soft_update_module(self.target_policy, self.policy, self.config.tau)
        soft_update_module(self.target_critic, self.critic, self.config.tau)
        _soft_update_buffers(self.target_policy, self.policy, self.config.tau)
        _soft_update_buffers(self.target_critic, self.critic, self.config.tau)

    def __repr__(self) -> str:
        return (
            f"ATOCLearner(n_agents={self.n_agents}, action_dim={self.action_dim}, "
            f"replay_size={len(self.replay)}, period={self.config.communication_period})"
        )


def _safe_batch_norm(norm: nn.BatchNorm1d, values: Tensor) -> Tensor:
    if norm.training and values.shape[0] == 1:
        return F.batch_norm(
            values,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            eps=norm.eps,
        )
    return norm(values)


def _normal_initialization(module: nn.Module) -> None:
    """Paper-specified normal weights with explicit fan-in variance scaling."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=layer.weight.shape[1] ** -0.5,
            )
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LSTM):
            for name, parameter in layer.named_parameters():
                if "weight" in name:
                    nn.init.normal_(
                        parameter,
                        mean=0.0,
                        std=parameter.shape[1] ** -0.5,
                    )
                else:
                    nn.init.zeros_(parameter)


def _soft_update_buffers(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-average BatchNorm statistics as part of the target-network state."""
    target_buffers = dict(target.named_buffers())
    for name, source_buffer in source.named_buffers():
        target_buffer = target_buffers[name]
        if source_buffer.is_floating_point():
            target_buffer.mul_(1.0 - tau).add_(source_buffer, alpha=tau)
        else:
            target_buffer.copy_(source_buffer)


__all__ = [
    "PAPER_SOURCE",
    "ATOCConfig",
    "ATOCGroupScheduler",
    "ATOCLearner",
    "ATOCOutput",
    "ATOCPolicy",
]
