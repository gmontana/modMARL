"""TarMAC's targeted communication policy and synchronous actor-critic.

Model: each shared GRU consumes a local observation and the targeted message from
the preceding timestep. Sender signatures/values and receiver queries define a
scaled dot-product attention message; optional extra communication rounds apply
the paper's recurrent message-state update before the categorical action head.
Invariants: attention rows are receivers and columns are senders; the critic sees
all agents' policy hidden states and executed actions only during training.
Interface: :class:`TarMACAgent` owns the policy, centralized Q critic, and one
complete-rollout update.
Why: this follows Das et al., ICML 2019, equations (1)--(4) and the paper's
experiment settings. The authors did not release an implementation. The
formerly cited WMG mirror is no longer publicly cloneable, so no unverifiable
fork behavior is treated as normative.

The paper trains a batched synchronous actor-critic with RMSProp (learning rate
7e-4, alpha 0.99), batch size 16, gamma 0.99, and policy entropy coefficient
0.01. The centralized critic estimates Q(h_1, ..., h_N, a_1, ..., a_N) by
temporal difference, and the shared stochastic policy uses that detached Q in
the stated policy gradient. There is no replay buffer, deterministic policy
gradient, Gumbel action relaxation, target network, or soft target update.
The paper's objective ends at its fixed episode horizon ``T``; the final TD
target therefore has no bootstrap at either an earlier terminal state or that
horizon boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.nn import build_mlp


@dataclass(frozen=True)
class TarMACConfig:
    """Architecture and optimization defaults reported by the TarMAC paper."""

    hidden_dim: int = 128
    message_dim: int = 32
    signature_dim: int = 16
    communication_rounds: int = 1
    learning_rate: float = 7e-4
    rmsprop_alpha: float = 0.99
    gamma: float = 0.99
    entropy_coefficient: float = 0.01


class TarMACPolicy(nn.Module):
    """Shared recurrent categorical policy with targeted delayed messages."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: TarMACConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TarMACConfig()
        cfg = self.config
        if cfg.communication_rounds < 1:
            raise ValueError("TarMAC requires at least one communication round")
        self.action_dim = action_dim
        self.hidden_dim = cfg.hidden_dim
        self.message_dim = cfg.message_dim
        self.scale = cfg.signature_dim ** -0.5
        self.gru = nn.GRUCell(obs_dim + cfg.message_dim, cfg.hidden_dim)
        self.query = nn.Linear(cfg.hidden_dim, cfg.signature_dim)
        self.signature = nn.Linear(cfg.hidden_dim, cfg.signature_dim)
        self.value = nn.Linear(cfg.hidden_dim, cfg.message_dim)
        self.round_update = nn.Linear(cfg.hidden_dim + cfg.message_dim, cfg.hidden_dim)
        self.action_head = nn.Linear(cfg.hidden_dim, action_dim)

    def initial_state(self, batch: int, n_agents: int, device: torch.device) -> Tensor:
        """Return packed hidden/message state ``(B, n, hidden + message)``."""
        return torch.zeros(
            batch,
            n_agents,
            self.hidden_dim + self.message_dim,
            device=device,
        )

    def _target(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        query = self.query(hidden)
        signature = self.signature(hidden)
        value = self.value(hidden)
        score = torch.matmul(query, signature.transpose(-1, -2)) * self.scale
        attention = torch.softmax(score, dim=-1)
        return torch.matmul(attention, value), attention

    def forward(
        self,
        obs: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return action logits, next packed state, and final attention matrix."""
        batch, n_agents, _ = obs.shape
        hidden, received = state.split([self.hidden_dim, self.message_dim], dim=-1)
        hidden = self.gru(
            torch.cat([obs, received], dim=-1).reshape(
                batch * n_agents, obs.shape[-1] + self.message_dim,
            ),
            hidden.reshape(batch * n_agents, self.hidden_dim),
        ).view(batch, n_agents, self.hidden_dim)

        outgoing, attention = self._target(hidden)
        for _ in range(1, self.config.communication_rounds):
            hidden = torch.tanh(self.round_update(torch.cat([hidden, outgoing], dim=-1)))
            outgoing, attention = self._target(hidden)
        logits = self.action_head(hidden)
        return logits, torch.cat([hidden, outgoing], dim=-1), attention

    def act(
        self,
        obs: Tensor,
        state: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample the paper's categorical action, without a Gumbel relaxation."""
        logits, next_state, attention = self(obs, state)
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            next_state,
            attention,
        )


class TarMACCritic(nn.Module):
    """Centralized Q(h_1, ..., h_N, a_1, ..., a_N) from the paper."""

    def __init__(self, n_agents: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        input_dim = n_agents * (hidden_dim + action_dim)
        self.net = build_mlp(input_dim, [hidden_dim, hidden_dim], 1)

    def forward(self, hidden: Tensor, action_one_hot: Tensor) -> Tensor:
        inputs = torch.cat(
            [hidden.reshape(hidden.shape[0], -1), action_one_hot.reshape(hidden.shape[0], -1)],
            dim=-1,
        )
        return self.net(inputs).squeeze(-1)


class TarMACAgent(nn.Module):
    """TarMAC policy, centralized critic, and paper actor-critic optimizer."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        config: TarMACConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TarMACConfig()
        self.action_dim = action_dim
        self.policy = TarMACPolicy(obs_dim, action_dim, self.config)
        self.critic = TarMACCritic(n_agents, self.config.hidden_dim, action_dim)
        self.optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.config.learning_rate,
            alpha=self.config.rmsprop_alpha,
        )

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        mask: Tensor,
        continuation: Tensor,
    ) -> TarMACUpdate:
        """Apply one synchronous on-policy TD actor-critic update.

        ``obs`` is ``(batch, time, agents, obs_dim)``; ``actions`` is
        ``(batch, time, agents)``; team rewards, mask, and continuation are
        ``(batch, time)``. Complete trajectories must be collected by the
        current policy.
        """
        if rewards.shape != mask.shape or continuation.shape != mask.shape:
            raise ValueError("TarMAC team rewards, masks, and continuations must match")
        batch_size, horizon, n_agents = actions.shape
        state = self.policy.initial_state(batch_size, n_agents, obs.device)
        log_probs, entropies, hidden_states = [], [], []
        for step in range(horizon):
            logits, state, _ = self.policy(obs[:, step], state)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs.append(distribution.log_prob(actions[:, step]))
            entropies.append(distribution.entropy())
            hidden_states.append(state[..., :self.config.hidden_dim])
        log_prob_tensor = torch.stack(log_probs, dim=1)
        entropy_tensor = torch.stack(entropies, dim=1)
        hidden_tensor = torch.stack(hidden_states, dim=1)
        one_hot = F.one_hot(actions.long(), self.action_dim).to(obs.dtype)

        flat_hidden = hidden_tensor.detach().reshape(
            batch_size * horizon, n_agents, self.config.hidden_dim,
        )
        flat_actions = one_hot.reshape(batch_size * horizon, n_agents, self.action_dim)
        q_values = self.critic(flat_hidden, flat_actions).view(batch_size, horizon)
        next_q = torch.cat([q_values[:, 1:].detach(), torch.zeros_like(q_values[:, :1])], dim=1)
        td_target = rewards + self.config.gamma * continuation * next_q
        denominator = mask.sum().clamp_min(1.0)
        critic_loss = ((q_values - td_target).square() * mask).sum() / denominator

        joint_log_prob = log_prob_tensor.sum(dim=-1)
        joint_entropy = entropy_tensor.sum(dim=-1)
        policy_loss = -(
            (joint_log_prob * q_values.detach() + self.config.entropy_coefficient * joint_entropy)
            * mask
        ).sum() / denominator
        loss = policy_loss + critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return TarMACUpdate(
            policy_loss=float(policy_loss.detach()),
            critic_loss=float(critic_loss.detach()),
            entropy=float((joint_entropy * mask).sum().detach() / denominator),
            q_mean=float((q_values * mask).sum().detach() / denominator),
        )


@dataclass(frozen=True)
class TarMACUpdate:
    """Diagnostics from one synchronous TarMAC actor-critic step."""

    policy_loss: float
    critic_loss: float
    entropy: float
    q_mean: float


__all__ = [
    "TarMACAgent",
    "TarMACConfig",
    "TarMACCritic",
    "TarMACPolicy",
    "TarMACUpdate",
]
