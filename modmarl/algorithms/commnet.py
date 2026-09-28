"""CommNet's released recurrent communication policy and REINFORCE learner.

Model: a shared LSTM receives each agent's local observation, its recurrent
hidden state, and the mean previous-step hidden state of the other active agents.
Shared categorical-policy and scalar-baseline heads read the new hidden state.
Invariants: sender self-edges are absent; communication is zero at episode start;
cooperative returns retain an agent axis but are averaged across the team.
Interface: :class:`CommNetAgent` owns recurrent acting and one complete-rollout
REINFORCE update.
Why: Sukhbaatar, Szlam, and Fergus, NeurIPS 2016, and the authors' release at
``facebookarchive/CommNet@3fc1fe801925bac3055d5b4730a7948649eead11``
define both the temporal message path and its learner.

The release's LSTM equation, 50-unit state, 0.1 initial hidden/cell state,
normal initialization with standard deviation 0.2, local learned baseline,
RMSProp at 1e-3 (alpha 0.97, effective epsilon 1e-8), baseline coefficient 0.03,
undiscounted returns, and cooperative reward averaging are retained. The paper
used 18 workers of 16 games (288 games per aggregate batch); the single-process
trainer therefore defaults to 288. Its default overlapping truncated BPTT gives
four loss steps ten preceding context steps, matching ``unroll=10`` and
``unroll_freq=4``. There is no deterministic actor-critic, replay, target network,
or Gumbel action relaxation.

The release declares ``rmsprop_eps=1e-6`` but passes the misspelled
``rmsprob_eps`` to Torch's optimizer, so training actually uses Torch's 1e-8
default. That executed behavior is retained here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CommNetCell(nn.Module):
    """One released LSTM-CommNet step with delayed other-agent communication."""

    def __init__(self, obs_dim: int, hidden_dim: int = 50) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_encoder = nn.Linear(obs_dim, 4 * hidden_dim)
        self.hidden_encoder = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.comm_encoder = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.2)
            nn.init.normal_(module.bias, std=0.2)

    @staticmethod
    def receive(hidden: Tensor, alive: Tensor) -> Tensor:
        """Average other active senders for every active receiver."""
        _batch, n_agents, _ = hidden.shape
        identity = torch.eye(n_agents, device=hidden.device, dtype=hidden.dtype)
        edges = alive.to(hidden.dtype).unsqueeze(2) * alive.to(hidden.dtype).unsqueeze(1)
        edges = edges * (1.0 - identity)
        denominator = edges.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(edges, hidden) / denominator

    def forward(
        self,
        obs: Tensor,
        hidden: Tensor,
        cell: Tensor,
        received: Tensor,
        alive: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return hidden, cell, and the message received on the next step."""
        preactivation = (
            self.obs_encoder(obs)
            + self.hidden_encoder(hidden)
            + self.comm_encoder(received)
        )
        forget, write, read, candidate = preactivation.chunk(4, dim=-1)
        new_cell = cell * torch.sigmoid(forget) + torch.tanh(candidate) * torch.sigmoid(write)
        new_hidden = torch.tanh(new_cell) * torch.sigmoid(read)
        return new_hidden, new_cell, self.receive(new_hidden, alive)


class CommNetActor(nn.Module):
    """Shared recurrent CommNet policy and learned local baseline."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 50) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = CommNetCell(obs_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.baseline_head = nn.Linear(hidden_dim, 1)
        CommNetCell._initialize(self.policy_head)
        CommNetCell._initialize(self.baseline_head)

    def initial_state(
        self,
        batch: int,
        n_agents: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return released 0.1 hidden/cell states and zero initial messages."""
        shape = (batch, n_agents, self.hidden_dim)
        hidden = torch.full(shape, 0.1, device=device)
        cell = torch.full(shape, 0.1, device=device)
        return hidden, cell, torch.zeros(shape, device=device)

    def step(
        self,
        obs: Tensor,
        hidden: Tensor,
        cell: Tensor,
        received: Tensor,
        alive: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return logits, baseline, and the next recurrent communication state."""
        hidden, cell, received = self.cell(obs, hidden, cell, received, alive)
        return (
            self.policy_head(hidden),
            self.baseline_head(hidden).squeeze(-1),
            hidden,
            cell,
            received,
        )


@dataclass(frozen=True)
class CommNetUpdate:
    """Diagnostics from one complete rollout update."""

    policy_loss: float
    baseline_loss: float
    return_mean: float


class CommNetAgent(nn.Module):
    """Released recurrent policy, baseline, and RMSProp learner."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 50,
        *,
        learning_rate: float = 1e-3,
        baseline_coefficient: float = 0.03,
        unroll_length: int = 10,
        unroll_frequency: int = 4,
    ) -> None:
        super().__init__()
        if unroll_length > 0 and unroll_frequency > unroll_length:
            raise ValueError("unroll_frequency cannot exceed unroll_length")
        self.actor = CommNetActor(obs_dim, action_dim, hidden_dim)
        self.baseline_coefficient = baseline_coefficient
        self.unroll_length = unroll_length
        self.unroll_frequency = unroll_frequency
        self.optimizer = torch.optim.RMSprop(
            self.actor.parameters(), lr=learning_rate, alpha=0.97, eps=1e-8,
        )

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        hidden: Tensor,
        cell: Tensor,
        received: Tensor,
        alive: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample categorical actions and advance recurrent communication."""
        logits, _, hidden, cell, received = self.actor.step(
            obs, hidden, cell, received, alive,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, hidden, cell, received

    def _states_before(self, obs: Tensor, mask: Tensor) -> list[tuple[Tensor, Tensor, Tensor]]:
        batch, horizon, n_agents, _ = obs.shape
        hidden, cell, received = self.actor.initial_state(batch, n_agents, obs.device)
        states = [(hidden, cell, received)]
        with torch.no_grad():
            for step in range(horizon):
                _, _, hidden, cell, received = self.actor.step(
                    obs[:, step], hidden, cell, received, mask[:, step],
                )
                states.append((hidden, cell, received))
        return states

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        mask: Tensor,
    ) -> CommNetUpdate:
        """Apply the released cooperative REINFORCE update.

        ``obs`` is ``(batch, time, agents, obs_dim)``; actions and mask are
        ``(batch, time, agents)``. A shared reward is ``(batch, time)``;
        agent-specific rewards are averaged before future returns are formed.
        """
        if rewards.ndim == actions.ndim:
            rewards = rewards.mean(dim=-1)
        if rewards.shape != actions.shape[:2]:
            raise ValueError("CommNet requires one shared reward per environment step")
        team_rewards = rewards.unsqueeze(-1).expand_as(mask)
        returns = torch.zeros_like(team_rewards)
        running = torch.zeros_like(team_rewards[:, 0])
        for step in range(rewards.shape[1] - 1, -1, -1):
            running = (team_rewards[:, step] + running) * mask[:, step]
            returns[:, step] = running

        states = self._states_before(obs, mask)
        horizon = obs.shape[1]
        frequency = horizon if self.unroll_length == 0 else self.unroll_frequency
        policy_loss = torch.zeros((), device=obs.device)
        baseline_loss = torch.zeros((), device=obs.device)
        for end in range(horizon, 0, -frequency):
            loss_start = max(0, end - frequency)
            context_start = 0 if self.unroll_length == 0 else max(0, end - self.unroll_length - 1)
            hidden, cell, received = (state.detach() for state in states[context_start])
            for step in range(context_start, end):
                logits, baseline, hidden, cell, received = self.actor.step(
                    obs[:, step], hidden, cell, received, mask[:, step],
                )
                if step < loss_start:
                    continue
                log_prob = F.log_softmax(logits, dim=-1).gather(
                    -1, actions[:, step].long().unsqueeze(-1),
                ).squeeze(-1)
                advantage = returns[:, step] - baseline.detach()
                policy_loss = policy_loss - (
                    log_prob * advantage * mask[:, step]
                ).sum()
                baseline_loss = baseline_loss + (
                    (baseline - returns[:, step]).square() * mask[:, step]
                ).sum()
        batch_size = obs.shape[0]
        policy_loss = policy_loss / batch_size
        baseline_loss = baseline_loss / batch_size
        loss = policy_loss + self.baseline_coefficient * baseline_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return CommNetUpdate(
            policy_loss=float(policy_loss.detach()),
            baseline_loss=float(baseline_loss.detach()),
            return_mean=float(returns[mask.bool()].mean()),
        )


__all__ = ["CommNetActor", "CommNetAgent", "CommNetCell", "CommNetUpdate"]
