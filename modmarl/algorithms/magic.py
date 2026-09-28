"""MAGIC's released graph-communication policy and on-policy learner.

Model: a shared observation encoder and LSTM produce one latent per agent. Two
communication rounds each apply the released differentiable graph scheduler and
modified graph-attention aggregation before shared policy and value heads.
Invariants: adjacency rows are receivers and columns are senders; every edge is a
hard Gumbel sample with a straight-through gradient; rewards keep their agent axis.
Interface: :class:`MAGICAgent` acts recurrently and applies complete-rollout
REINFORCE updates through :meth:`MAGICAgent.update`.
Why: the AAMAS 2021 paper and the pinned authors' release define the graph model;
revision ``0ad3a6126f46e475f8d46ab61e67fddbfe99e7d8`` defines underspecified
architecture and optimization details.

The released learner is retained exactly: individual returns by default
(``mean_ratio=0``), a learned value baseline, RMSProp at 1e-3 (alpha 0.97,
epsilon 1e-6), gamma 1, value coefficient 0.015, no entropy bonus, gradients
divided by the number of environment transitions, and recurrent detachment every
10 steps. It is not replaced by PPO, GAE, Adam, or replay.

The default architecture is the released predator-prey-medium configuration:
128 recurrent units, directed independently learned graphs in both rounds, a
4-head 32-unit first GAT, normalized attention, a complete-graph GAT encoder for
the schedulers, and learned self-loops. Other released task settings are expressed
through :class:`MAGICConfig`; no alternative communication equation is hidden in
the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.nn import build_mlp


class SelfLoopMode(IntEnum):
    """Released GAT self-loop controls."""

    WITHOUT = 0
    WITH = 1
    LEARNED = 2


@dataclass(frozen=True)
class MAGICConfig:
    """Released predator-prey-medium architecture and optimization settings."""

    hidden_dim: int = 128
    gat_hidden_dim: int = 32
    gat_heads: int = 4
    directed: bool = True
    use_gat_encoder: bool = True
    gat_encoder_dim: int = 32
    gat_encoder_heads: int = 8
    learn_second_graph: bool = True
    first_graph_complete: bool = False
    second_graph_complete: bool = False
    first_normalize: bool = True
    second_normalize: bool = True
    encoder_normalize: bool = False
    first_self_loop: SelfLoopMode = SelfLoopMode.LEARNED
    second_self_loop: SelfLoopMode = SelfLoopMode.LEARNED
    message_encoder: bool = False
    message_decoder: bool = False
    learning_rate: float = 1e-3
    gamma: float = 1.0
    mean_ratio: float = 0.0
    value_coefficient: float = 0.015
    entropy_coefficient: float = 0.0
    detach_gap: int = 10


class GraphAttentionScheduler(nn.Module):
    """Released hard pairwise scheduler with optional undirected logits."""

    def __init__(
        self,
        dim: int,
        *,
        directed: bool = True,
        second_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.directed = directed
        second_hidden = dim // 8 if second_hidden_dim is None else second_hidden_dim
        self.mlp = build_mlp(2 * dim, [dim // 2, second_hidden], 2)

    def forward(
        self,
        features: Tensor,
        alive: Tensor | None = None,
        *,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return adjacency ``(B, receiver, sender)`` and sampled Gumbel noise."""
        batch, n_agents, _ = features.shape
        receiver = features.unsqueeze(2).expand(-1, -1, n_agents, -1)
        sender = features.unsqueeze(1).expand(-1, n_agents, -1, -1)
        pair = torch.cat([receiver, sender], dim=-1)
        logits = self.mlp(pair)
        if not self.directed:
            reverse = self.mlp(pair.transpose(1, 2))
            logits = 0.5 * logits + 0.5 * reverse
        if noise is None:
            uniform = torch.rand_like(logits).clamp_(1e-20, 1.0 - 1e-7)
            noise = -torch.log(-torch.log(uniform))
        soft = torch.softmax(logits + noise, dim=-1)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
        adjacency = (hard - soft.detach() + soft)[..., 1]
        if alive is None:
            alive = torch.ones(batch, n_agents, device=features.device, dtype=features.dtype)
        edge_mask = alive.to(features.dtype).unsqueeze(2) * alive.to(features.dtype).unsqueeze(1)
        return adjacency * edge_mask, noise


class GATMessageProcessor(nn.Module):
    """Released differentiable graph-attention equation, batched over episodes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_heads: int = 1,
        average: bool = False,
        normalize: bool = False,
        self_loop: SelfLoopMode = SelfLoopMode.LEARNED,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.average = average
        self.normalize = normalize
        self.self_loop = self_loop
        self.weight = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        self.attn_receiver = nn.Parameter(torch.empty(num_heads, out_dim))
        self.attn_sender = nn.Parameter(torch.empty(num_heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim if average else num_heads * out_dim))
        gain = nn.init.calculate_gain("relu")
        projection = self.weight.new_empty(in_dim, num_heads * out_dim)
        nn.init.xavier_normal_(projection, gain=gain)
        with torch.no_grad():
            self.weight.copy_(projection.view(in_dim, num_heads, out_dim).permute(1, 0, 2))
        nn.init.xavier_normal_(self.attn_receiver.unsqueeze(-1), gain=gain)
        nn.init.xavier_normal_(self.attn_sender.unsqueeze(-1), gain=gain)

    def forward(self, messages: Tensor, adjacency: Tensor) -> Tensor:
        """Aggregate sender messages into receiver rows."""
        batch, n_agents, _ = messages.shape
        identity = torch.eye(n_agents, device=messages.device, dtype=messages.dtype).expand(batch, -1, -1)
        if self.self_loop is SelfLoopMode.WITHOUT:
            adjacency = adjacency * (1.0 - identity)
        elif self.self_loop is SelfLoopMode.WITH:
            adjacency = identity + adjacency * (1.0 - identity)

        transformed = torch.einsum("bnd,hde->bhne", messages, self.weight)
        receiver = (transformed * self.attn_receiver.view(1, self.num_heads, 1, -1)).sum(-1)
        sender = (transformed * self.attn_sender.view(1, self.num_heads, 1, -1)).sum(-1)
        scores = F.leaky_relu(receiver.unsqueeze(-1) + sender.unsqueeze(-2), 0.2)
        edges = adjacency.unsqueeze(1)
        attention = torch.softmax(scores * edges, dim=-1) * edges
        if self.normalize:
            if self.self_loop is not SelfLoopMode.WITH:
                attention = attention + 1e-15
            attention = attention / attention.sum(dim=-1, keepdim=True)
            attention = attention * edges
        aggregated = torch.matmul(attention, transformed)
        if self.average:
            output = aggregated.mean(dim=1)
        else:
            output = aggregated.permute(0, 2, 1, 3).reshape(batch, n_agents, -1)
        return output + self.bias


class MAGICAgent(nn.Module):
    """MAGIC recurrent policy, graph communicator, baseline, and released learner."""

    def __init__(self, obs_dim: int, action_dim: int, config: MAGICConfig | None = None) -> None:
        super().__init__()
        self.config = config or MAGICConfig()
        cfg = self.config
        self.hidden_dim = cfg.hidden_dim
        self.encoder = nn.Linear(obs_dim, cfg.hidden_dim)
        self.lstm = nn.LSTMCell(cfg.hidden_dim, cfg.hidden_dim)
        self.message_encoder = nn.Linear(cfg.hidden_dim, cfg.hidden_dim) if cfg.message_encoder else nn.Identity()
        self.message_decoder = nn.Linear(cfg.hidden_dim, cfg.hidden_dim) if cfg.message_decoder else nn.Identity()

        if cfg.use_gat_encoder:
            self.gat_encoder = GATMessageProcessor(
                cfg.hidden_dim,
                cfg.gat_encoder_dim,
                num_heads=cfg.gat_encoder_heads,
                average=True,
                normalize=cfg.encoder_normalize,
                self_loop=SelfLoopMode.WITH,
            )
            scheduler_dim = cfg.gat_encoder_dim
        else:
            self.gat_encoder = None
            scheduler_dim = cfg.hidden_dim
        scheduler_second = scheduler_dim // 2 if cfg.use_gat_encoder else scheduler_dim // 8
        self.first_scheduler = None if cfg.first_graph_complete else GraphAttentionScheduler(
            scheduler_dim,
            directed=cfg.directed,
            second_hidden_dim=scheduler_second,
        )
        self.second_scheduler = (
            GraphAttentionScheduler(
                scheduler_dim,
                directed=cfg.directed,
                second_hidden_dim=scheduler_second,
            )
            if cfg.learn_second_graph and not cfg.second_graph_complete
            else None
        )
        self.first_processor = GATMessageProcessor(
            cfg.hidden_dim,
            cfg.gat_hidden_dim,
            num_heads=cfg.gat_heads,
            average=False,
            normalize=cfg.first_normalize,
            self_loop=cfg.first_self_loop,
        )
        self.second_processor = GATMessageProcessor(
            cfg.gat_hidden_dim * cfg.gat_heads,
            cfg.hidden_dim,
            average=True,
            normalize=cfg.second_normalize,
            self_loop=cfg.second_self_loop,
        )
        self.policy_head = nn.Linear(2 * cfg.hidden_dim, action_dim)
        self.value_head = nn.Linear(2 * cfg.hidden_dim, 1)
        self.optimizer = torch.optim.RMSprop(
            self.parameters(), lr=cfg.learning_rate, alpha=0.97, eps=1e-6,
        )

    def init_state(self, batch: int, n_agents: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return zero recurrent states ``(B, n, hidden_dim)``."""
        shape = (batch, n_agents, self.hidden_dim)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @staticmethod
    def _complete_graph(alive: Tensor) -> Tensor:
        alive = alive.to(dtype=torch.get_default_dtype())
        return alive.unsqueeze(2) * alive.unsqueeze(1)

    def forward(
        self,
        obs: Tensor,
        hidden: Tensor,
        cell: Tensor,
        *,
        alive: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one policy step; noise has shape ``(2, B, n, n, 2)`` when replayed."""
        batch, n_agents, _ = obs.shape
        if alive is None:
            alive = torch.ones(batch, n_agents, device=obs.device, dtype=obs.dtype)
        encoded = self.encoder(obs).reshape(batch * n_agents, self.hidden_dim)
        new_hidden, new_cell = self.lstm(
            encoded,
            (
                hidden.reshape(batch * n_agents, self.hidden_dim),
                cell.reshape(batch * n_agents, self.hidden_dim),
            ),
        )
        new_hidden = new_hidden.view(batch, n_agents, self.hidden_dim)
        new_cell = new_cell.view(batch, n_agents, self.hidden_dim)
        original_message = self.message_encoder(new_hidden) * alive.unsqueeze(-1)

        complete = self._complete_graph(alive).to(obs.dtype)
        scheduler_features = (
            self.gat_encoder(original_message, complete)
            if self.gat_encoder is not None
            else original_message
        )
        noises: list[Tensor] = []
        if self.first_scheduler is None:
            first_adjacency = complete
            first_noise = torch.zeros(batch, n_agents, n_agents, 2, device=obs.device, dtype=obs.dtype)
        else:
            first_replay = None if noise is None else noise[0]
            first_adjacency, first_noise = self.first_scheduler(
                scheduler_features, alive, noise=first_replay,
            )
        noises.append(first_noise)
        message = F.elu(self.first_processor(original_message, first_adjacency))

        if self.config.second_graph_complete:
            second_adjacency = complete
            second_noise = torch.zeros_like(first_noise)
        elif self.second_scheduler is None:
            second_adjacency = first_adjacency
            second_noise = first_noise
        else:
            second_replay = None if noise is None else noise[1]
            second_adjacency, second_noise = self.second_scheduler(
                scheduler_features, alive, noise=second_replay,
            )
        noises.append(second_noise)
        message = self.second_processor(message, second_adjacency) * alive.unsqueeze(-1)
        message = self.message_decoder(message)

        features = torch.cat([new_hidden, message], dim=-1)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        adjacency = torch.stack([first_adjacency, second_adjacency])
        return logits, value, new_hidden, new_cell, adjacency, torch.stack(noises)

    def act(
        self,
        obs: Tensor,
        hidden: Tensor,
        cell: Tensor,
        *,
        alive: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample one environment action and expose the realized graph."""
        logits, value, new_hidden, new_cell, adjacency, noise = self(
            obs, hidden, cell, alive=alive,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value, new_hidden, new_cell, adjacency, noise

    def evaluate_step(
        self,
        obs: Tensor,
        hidden: Tensor,
        cell: Tensor,
        action: Tensor,
        noise: Tensor,
        *,
        alive: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Re-evaluate a collected action under its original Gumbel perturbation."""
        logits, value, new_hidden, new_cell, _, _ = self(
            obs, hidden, cell, alive=alive, noise=noise,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value, new_hidden, new_cell

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        mask: Tensor,
        noise: Tensor,
        *,
        alive: Tensor | None = None,
        continuation: Tensor | None = None,
    ) -> MAGICUpdate:
        """Apply one released rollout update.

        ``obs`` is ``(batch, time, agents, obs_dim)``; actions, rewards, mask,
        alive, and continuation are ``(batch, time, agents)``; noise is
        ``(batch, time, 2, agents, agents, 2)``.
        """
        if rewards.shape != actions.shape or mask.shape != actions.shape:
            raise ValueError("MAGIC requires per-agent rewards and masks")
        if alive is None:
            alive = mask
        if continuation is None:
            continuation = mask
        batch_size, horizon, n_agents = actions.shape
        cooperative = torch.zeros_like(rewards)
        individual = torch.zeros_like(rewards)
        team_running = torch.zeros_like(rewards[:, 0])
        individual_running = torch.zeros_like(rewards[:, 0])
        for step in range(horizon - 1, -1, -1):
            valid = mask[:, step]
            team_running = (rewards[:, step] + self.config.gamma * team_running) * valid
            individual_running = (
                rewards[:, step]
                + self.config.gamma * individual_running * continuation[:, step]
            ) * valid
            cooperative[:, step] = team_running
            individual[:, step] = individual_running
        returns = (
            self.config.mean_ratio * cooperative.mean(dim=-1, keepdim=True)
            + (1.0 - self.config.mean_ratio) * individual
        )

        hidden, cell = self.init_state(batch_size, n_agents, obs.device)
        log_probs, entropies, values = [], [], []
        for step in range(horizon):
            step_noise = noise[:, step].permute(1, 0, 2, 3, 4)
            log_prob, entropy, value, hidden, cell = self.evaluate_step(
                obs[:, step],
                hidden,
                cell,
                actions[:, step],
                step_noise,
                alive=alive[:, step],
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            if (step + 1) % self.config.detach_gap == 0:
                hidden, cell = hidden.detach(), cell.detach()
        log_prob_tensor = torch.stack(log_probs, dim=1)
        entropy_tensor = torch.stack(entropies, dim=1)
        value_tensor = torch.stack(values, dim=1)

        loss_mask = mask * alive
        advantage = returns - value_tensor.detach()
        transition_count = mask.any(dim=-1).sum().clamp_min(1).to(obs.dtype)
        action_loss = -(log_prob_tensor * advantage * loss_mask).sum() / transition_count
        value_loss = ((value_tensor - returns).square() * loss_mask).sum() / transition_count
        entropy = (entropy_tensor * loss_mask).sum() / transition_count
        loss = action_loss + self.config.value_coefficient * value_loss
        loss = loss - self.config.entropy_coefficient * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return MAGICUpdate(
            action_loss=float(action_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropy.detach()),
            return_mean=float(returns[loss_mask.bool()].mean()),
        )


@dataclass(frozen=True)
class MAGICUpdate:
    """Diagnostics from one released-style MAGIC update."""

    action_loss: float
    value_loss: float
    entropy: float
    return_mean: float


__all__ = [
    "GATMessageProcessor",
    "GraphAttentionScheduler",
    "MAGICAgent",
    "MAGICConfig",
    "MAGICUpdate",
    "SelfLoopMode",
]
