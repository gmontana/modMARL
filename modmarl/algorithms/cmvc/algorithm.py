"""Counterfactual Message Value Communication (CMVC).

Model: a parameter-shared GRU encodes each local trajectory; a local selector predicts
one directed counterfactual message value (CMV) per possible sender and requests exactly
the positive ones.  Requested sender embeddings enter a two-layer monotone mixing network
whose nonnegative weights are generated from the receiver's rectified CMV vector, then a
shared discrete MADDPG actor chooses the action.  A centered parameter-shared centralized
critic evaluates every receiver from the joint observations and actions.
Invariants: self requests and self selector loss are always zero, a missing sender is the
all-zero action segment in Eq. 1, selector labels are detached critic differences, the hard
request has no policy-gradient path, and every mixing weight is nonnegative so
``d message / d received_embedding >= 0`` elementwise.
Interface: ``CMVCPolicy`` owns recurrent execution, ``CMVCCritic.counterfactual_values``
owns Eq. 1, and ``CMVCLearner`` owns episode replay plus the separate Eqs. 5--9 updates.
Why: Gao et al., *Communication in Multiagent Reinforcement Learning via Counterfactual
Message Value*, IEEE TSMC: Systems 55(11), 2025, DOI 10.1109/TSMC.2025.3604230.  The
paper-linked ``GaoZiHong/CMVC`` repository exists but is empty, so no code revision can be
pinned.  The paper leaves the selector depth, mixing-network width, critic's fixed-width
absent-action encoding, communication warmup, and target interval underspecified; this
module uses a two-layer 64-unit selector and 32-unit mixer, and exposes the zero-segment
contract, 100-update warmup, and 200-update hard target interval explicitly.  The published
70 threshold is represented as the 70th batch percentile.  Validation is a bounded version
of cooperative navigation, not the paper's 400,000-episode nine-agent experiment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...common.replay import EpisodeBatch, EpisodeReplayBuffer
from ...components import gumbel_policy_sample

PAPER_SOURCE = "paper:doi-10.1109/TSMC.2025.3604230;repository:GaoZiHong/CMVC-empty"


@dataclass(frozen=True)
class CMVCConfig:
    """Cooperative-navigation settings and explicit paper-underspecified schedules."""

    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 128
    hyper_hidden_dim: int = 32
    message_hidden_dim: int = 32
    gamma: float = 0.95
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-2
    selector_learning_rate: float = 1e-3
    replay_capacity: int = 5_000
    batch_size: int = 32
    communication_warmup_updates: int = 100
    target_update_interval: int = 200
    pruning_percentile: float = 70.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.pruning_percentile <= 100.0:
            raise ValueError("pruning_percentile must be in [0, 100]")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be positive")


@dataclass(frozen=True)
class CMVCPolicyOutput:
    action_vectors: Tensor
    action_indices: Tensor
    logits: Tensor
    trajectory_embeddings: Tensor
    cmv_predictions: Tensor
    request_gates: Tensor
    messages: Tensor
    hidden: Tensor


@dataclass(frozen=True)
class CMVCUpdate:
    critic_loss: float
    actor_loss: float
    selector_loss: float
    target_q_mean: float
    communication_rate: float


class CMVCMonotonicAggregator(nn.Module):
    """CMV-conditioned hypernetwork mixer satisfying the paper's Eq. 4."""

    def __init__(
        self,
        n_agents: int,
        embedding_dim: int,
        message_dim: int,
        hyper_hidden_dim: int = 32,
        mixing_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim
        flat_input = n_agents * embedding_dim
        self.first_weights = _hypernetwork(
            n_agents, hyper_hidden_dim, flat_input * mixing_hidden_dim,
        )
        self.first_bias = _hypernetwork(n_agents, hyper_hidden_dim, mixing_hidden_dim)
        self.second_weights = _hypernetwork(
            n_agents, hyper_hidden_dim, mixing_hidden_dim * message_dim,
        )
        self.final_bias = _hypernetwork(n_agents, hyper_hidden_dim, message_dim)
        self.mixing_hidden_dim = mixing_hidden_dim

    def forward(self, sender_embeddings: Tensor, cmv_predictions: Tensor) -> tuple[Tensor, Tensor]:
        """Aggregate ``(B,N,H)`` senders for every receiver from ``(B,N,N)`` CMVs."""
        batch, n_agents, _ = sender_embeddings.shape
        if n_agents != self.n_agents or cmv_predictions.shape != (batch, n_agents, n_agents):
            raise ValueError("sender embeddings and CMVs must use the configured agent axes")
        off_diagonal = ~torch.eye(n_agents, dtype=torch.bool, device=sender_embeddings.device)
        gates = (cmv_predictions > 0) & off_diagonal.unsqueeze(0)
        received = gates.unsqueeze(-1) * sender_embeddings.unsqueeze(1)
        flat_received = received.reshape(batch * n_agents, -1)
        condition = F.relu(cmv_predictions).reshape(batch * n_agents, n_agents)

        first_weights = self.first_weights(condition).abs().reshape(
            batch * n_agents,
            self.n_agents * self.embedding_dim,
            self.mixing_hidden_dim,
        )
        first_bias = self.first_bias(condition).reshape(batch * n_agents, 1, -1)
        hidden = F.relu(torch.bmm(flat_received.unsqueeze(1), first_weights) + first_bias)
        second_weights = self.second_weights(condition).abs().reshape(
            batch * n_agents,
            self.mixing_hidden_dim,
            self.message_dim,
        )
        final_bias = F.relu(self.final_bias(condition)).reshape(batch * n_agents, 1, -1)
        message = torch.bmm(hidden, second_weights) + final_bias
        message = message.reshape(batch, n_agents, self.message_dim)
        message = message * gates.any(dim=-1, keepdim=True)
        return message, gates


class CMVCPolicy(nn.Module):
    """Shared recurrent encoder, selector, monotone aggregator, and local actor."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        config: CMVCConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or CMVCConfig()
        hidden = self.config.actor_hidden_dim
        self.observation_encoder = nn.Linear(obs_dim, hidden)
        self.trajectory_gru = nn.GRUCell(hidden, hidden)
        self.message_selector = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )
        self.message_aggregator = CMVCMonotonicAggregator(
            n_agents,
            hidden,
            hidden,
            self.config.hyper_hidden_dim,
            self.config.message_hidden_dim,
        )
        self.policy = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(
            batch_size,
            self.n_agents,
            self.config.actor_hidden_dim,
            device=device,
        )

    def step(
        self,
        obs: Tensor,
        hidden: Tensor,
        *,
        communication_enabled: bool = True,
        temperature: float = 1.0,
        hard: bool = True,
        deterministic: bool = False,
    ) -> CMVCPolicyOutput:
        """Execute one recurrent step on ``obs: (B,N,O)``."""
        batch, n_agents, _ = obs.shape
        encoded = F.relu(self.observation_encoder(obs))
        next_hidden = self.trajectory_gru(
            encoded.reshape(batch * n_agents, -1),
            hidden.reshape(batch * n_agents, -1),
        ).reshape(batch, n_agents, -1)
        predictions = self.message_selector(obs)
        diagonal = torch.eye(n_agents, dtype=torch.bool, device=obs.device).unsqueeze(0)
        predictions = predictions.masked_fill(diagonal, 0.0)
        communication_values = predictions if communication_enabled else torch.zeros_like(predictions)
        messages, gates = self.message_aggregator(next_hidden, communication_values.detach())
        logits = self.policy(torch.cat([next_hidden, messages], dim=-1))
        vectors, indices, _ = gumbel_policy_sample(
            logits,
            action_dim=self.action_dim,
            temperature=temperature,
            hard=hard,
            deterministic=deterministic,
        )
        return CMVCPolicyOutput(
            action_vectors=vectors,
            action_indices=indices,
            logits=logits,
            trajectory_embeddings=next_hidden,
            cmv_predictions=predictions,
            request_gates=gates,
            messages=messages,
            hidden=next_hidden,
        )

    def unroll(
        self,
        obs: Tensor,
        *,
        communication_enabled: bool = True,
        temperature: float = 1.0,
        hard: bool = True,
        deterministic: bool = False,
    ) -> CMVCPolicyOutput:
        """Unroll ``obs: (B,T,N,O)`` from a zero recurrent state."""
        hidden = self.initial_hidden(obs.shape[0], obs.device)
        outputs: list[CMVCPolicyOutput] = []
        for time in range(obs.shape[1]):
            output = self.step(
                obs[:, time],
                hidden,
                communication_enabled=communication_enabled,
                temperature=temperature,
                hard=hard,
                deterministic=deterministic,
            )
            hidden = output.hidden
            outputs.append(output)
        return CMVCPolicyOutput(
            action_vectors=torch.stack([output.action_vectors for output in outputs], dim=1),
            action_indices=torch.stack([output.action_indices for output in outputs], dim=1),
            logits=torch.stack([output.logits for output in outputs], dim=1),
            trajectory_embeddings=torch.stack(
                [output.trajectory_embeddings for output in outputs], dim=1,
            ),
            cmv_predictions=torch.stack([output.cmv_predictions for output in outputs], dim=1),
            request_gates=torch.stack([output.request_gates for output in outputs], dim=1),
            messages=torch.stack([output.messages for output in outputs], dim=1),
            hidden=hidden,
        )


class CMVCCritic(nn.Module):
    """Shared centralized critic with receiver-centered joint inputs."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        receiver_order = []
        for receiver in range(n_agents):
            receiver_order.append([receiver, *[peer for peer in range(n_agents) if peer != receiver]])
        self.register_buffer("receiver_order", torch.tensor(receiver_order, dtype=torch.long))
        joint_dim = n_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        """Return all receiver values from ``obs/actions: (B,N,*)``."""
        centered_obs = obs[:, self.receiver_order]
        centered_actions = actions[:, self.receiver_order]
        joint = torch.cat(
            [
                centered_obs.reshape(obs.shape[0], self.n_agents, -1),
                centered_actions.reshape(actions.shape[0], self.n_agents, -1),
            ],
            dim=-1,
        )
        return self.net(joint).squeeze(-1)

    def counterfactual_values(self, obs: Tensor, actions: Tensor) -> Tensor:
        """Compute Eq. 1 as ``Q_i(all actions) - Q_i(action_j absent)``."""
        full = self(obs, actions)
        sender_values = []
        for sender in range(self.n_agents):
            absent = actions.clone()
            absent[:, sender] = 0.0
            sender_values.append(full - self(obs, absent))
        values = torch.stack(sender_values, dim=-1)
        diagonal = torch.eye(self.n_agents, dtype=torch.bool, device=obs.device).unsqueeze(0)
        return values.masked_fill(diagonal, 0.0)


class CMVCLearner(nn.Module):
    """Complete CMVC learner with recurrent episode replay."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        config: CMVCConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.config = config or CMVCConfig()
        self.policy = CMVCPolicy(n_agents, obs_dim, action_dim, self.config)
        self.critic = CMVCCritic(
            n_agents, obs_dim, action_dim, self.config.critic_hidden_dim,
        )
        self.target_policy = copy.deepcopy(self.policy).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        selector_ids = {id(parameter) for parameter in self.policy.message_selector.parameters()}
        actor_parameters = [
            parameter for parameter in self.policy.parameters() if id(parameter) not in selector_ids
        ]
        self.actor_optimizer = torch.optim.Adam(
            actor_parameters, lr=self.config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate,
        )
        self.selector_optimizer = torch.optim.Adam(
            self.policy.message_selector.parameters(),
            lr=self.config.selector_learning_rate,
        )
        self.replay = EpisodeReplayBuffer(
            self.config.replay_capacity,
            horizon,
            n_agents,
            obs_dim,
        )
        self.update_count = 0

    @property
    def communication_enabled(self) -> bool:
        return self.update_count >= self.config.communication_warmup_updates

    def update(self, batch: EpisodeBatch | None = None) -> CMVCUpdate | None:
        """Apply critic, policy/encoder/aggregator, selector, and hard-target updates."""
        if batch is None:
            if len(self.replay) < self.config.batch_size:
                return None
            batch = self.replay.sample(self.config.batch_size, next(self.parameters()).device)
        batch_size, horizon, n_agents = batch.actions.shape
        valid = batch.mask.reshape(-1)
        obs = batch.obs[:, :-1].reshape(-1, n_agents, batch.obs.shape[-1])
        all_obs = batch.obs.reshape(batch_size, horizon + 1, n_agents, -1)
        replay_actions = F.one_hot(batch.actions, num_classes=self.action_dim).to(torch.float32)
        replay_actions = replay_actions.reshape(-1, n_agents, self.action_dim)

        with torch.no_grad():
            target_output = self.target_policy.unroll(
                all_obs,
                communication_enabled=self.communication_enabled,
                hard=False,
            )
            next_actions = target_output.action_vectors[:, 1:].reshape(
                -1, n_agents, self.action_dim,
            )
            next_obs = batch.obs[:, 1:].reshape(-1, n_agents, batch.obs.shape[-1])
            target_q = batch.rewards.reshape(-1, 1) + self.config.gamma * (
                1.0 - batch.dones.reshape(-1, 1)
            ) * self.target_critic(next_obs, next_actions)
        critic_q = self.critic(obs, replay_actions)
        critic_error = (critic_q - target_q).square() * valid.unsqueeze(-1)
        critic_loss = critic_error.sum() / valid.sum().clamp_min(1.0) / n_agents
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_output = self.policy.unroll(
            batch.obs[:, :-1],
            communication_enabled=self.communication_enabled,
            hard=False,
        )
        policy_actions = policy_output.action_vectors.reshape(-1, n_agents, self.action_dim)
        self.critic.requires_grad_(False)
        actor_terms = []
        for receiver in range(n_agents):
            counterfactual_actions = replay_actions.clone()
            counterfactual_actions[:, receiver] = policy_actions[:, receiver]
            actor_terms.append(self.critic(obs, counterfactual_actions)[:, receiver])
        actor_values = torch.stack(actor_terms, dim=-1)
        actor_loss = -(actor_values * valid.unsqueeze(-1)).sum() / (
            valid.sum().clamp_min(1.0) * n_agents
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        with torch.no_grad():
            true_values = self.critic.counterfactual_values(obs, replay_actions)
            selector_labels = shift_cmv_labels(
                true_values,
                valid,
                self.config.pruning_percentile,
            )
        predictions = policy_output.cmv_predictions.reshape(-1, n_agents, n_agents)
        off_diagonal = ~torch.eye(n_agents, dtype=torch.bool, device=obs.device)
        selector_mask = valid[:, None, None] * off_diagonal[None]
        selector_loss = ((predictions - selector_labels).square() * selector_mask).sum()
        selector_loss = selector_loss / selector_mask.sum().clamp_min(1.0)
        self.selector_optimizer.zero_grad(set_to_none=True)
        selector_loss.backward()
        self.selector_optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        request_gates = policy_output.request_gates.to(dtype=torch.float32)
        communication_rate = float(
            (request_gates * batch.mask[..., None, None]).sum()
            / (batch.mask.sum().clamp_min(1.0) * n_agents * max(n_agents - 1, 1))
        )
        return CMVCUpdate(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            selector_loss=float(selector_loss.detach()),
            target_q_mean=float((target_q * valid.unsqueeze(-1)).sum() / valid.sum().clamp_min(1.0) / n_agents),
            communication_rate=communication_rate,
        )

    def __repr__(self) -> str:
        return (
            f"CMVCLearner(n_agents={self.n_agents}, action_dim={self.action_dim}, "
            f"updates={self.update_count}, communication_enabled={self.communication_enabled})"
        )


def shift_cmv_labels(values: Tensor, valid: Tensor, percentile: float) -> Tensor:
    """Apply the paper's sorted-batch ``zeta(delta, B)`` threshold to Eq. 1 labels."""
    n_agents = values.shape[-1]
    off_diagonal = ~torch.eye(n_agents, dtype=torch.bool, device=values.device)
    valid_entries = valid.to(dtype=torch.bool)[:, None, None].expand(-1, n_agents, n_agents)
    selected = values[valid_entries & off_diagonal]
    if selected.numel() == 0:
        return torch.zeros_like(values)
    threshold = torch.quantile(selected, percentile / 100.0)
    shifted = values - threshold
    diagonal = torch.eye(n_agents, dtype=torch.bool, device=values.device).unsqueeze(0)
    return shifted.masked_fill(diagonal, 0.0)


def _hypernetwork(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


__all__ = [
    "PAPER_SOURCE",
    "CMVCConfig",
    "CMVCCritic",
    "CMVCLearner",
    "CMVCPolicy",
]
