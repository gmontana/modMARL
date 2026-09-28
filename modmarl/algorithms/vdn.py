"""Value-Decomposition Networks as specified by Sunehag et al. (AAMAS 2018).

Paper: arXiv:1706.05296v1, especially Sections 2.2, 3, and 4.1. Reference
implementations consulted: ``Louiii/ValueDecomposition@6067b1a`` and
``oxwhirl/pymarl@c971afd``; neither is the authors' original unpublished code.

Model: one parameter-shared recurrent dueling Q-network receives each agent's
local observation and one-hot role. Its selected utilities are added to form
``Q_tot``. Whole episodes are replayed, recurrence is detached every eight
steps, and the joint team reward trains the sum with the paper's truncated
forward-view lambda return (lambda=0.9). Execution is decentralized.

Invariants: no global state enters an agent utility; the only cross-agent
operation is the training-time sum; targets are hard copies; all defining
networks, exploration, return construction, optimizer, and updates live here.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.replay import EpisodeBatch


class VDNDuelingLSTM(nn.Module):
    """Paper network: 32-unit encoder, LSTM, and dueling value/advantage head."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self, inputs: Tensor, hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        encoded = self.encoder(inputs)
        recurrent, hidden = self.lstm(encoded, hidden)
        value = self.value(recurrent)
        advantage = self.advantage(recurrent)
        return value + advantage - advantage.mean(dim=-1, keepdim=True), hidden


class VDNAgent(nn.Module):
    """Complete recurrent VDN learner and decentralized execution policy."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        *,
        role_information: bool = True,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        trace_lambda: float = 0.9,
        trace_length: int = 8,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_anneal_steps: int = 50_000,
    ) -> None:
        super().__init__()
        if n_agents < 1 or trace_length < 1 or target_update_interval < 1:
            raise ValueError("agent count and update intervals must be positive")
        if not 0.0 <= trace_lambda <= 1.0:
            raise ValueError("trace_lambda must lie in [0, 1]")
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.role_information = role_information
        self.gamma = gamma
        self.trace_lambda = trace_lambda
        self.trace_length = trace_length
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_anneal_steps = epsilon_anneal_steps
        input_dim = obs_dim + (n_agents if role_information else 0)
        self.q_network = VDNDuelingLSTM(input_dim, action_dim, hidden_dim)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self._execution_hidden: tuple[Tensor, Tensor] | None = None

    def epsilon(self, env_step: int) -> float:
        fraction = min(max(env_step, 0) / self.epsilon_anneal_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def reset_hidden(self) -> None:
        """Reset decentralized recurrent state at an episode boundary."""
        self._execution_hidden = None

    @torch.no_grad()
    def act(self, obs: Tensor, epsilon: float) -> Tensor:
        """Select actions from local ``obs (n_agents, obs_dim)`` and advance recurrence."""
        if obs.shape[0] != self.n_agents:
            raise ValueError(f"expected {self.n_agents} observations, got {obs.shape[0]}")
        inputs = self._append_roles(obs.view(1, 1, self.n_agents, -1))
        flat = inputs.reshape(self.n_agents, 1, -1)
        q_values, self._execution_hidden = self.q_network(flat, self._execution_hidden)
        q_values = q_values[:, 0]
        greedy = q_values.argmax(dim=-1)
        random_actions = torch.randint(self.action_dim, (self.n_agents,), device=obs.device)
        explore = torch.rand(self.n_agents, device=obs.device) < epsilon
        return torch.where(explore, random_actions, greedy)

    def update(self, batch: EpisodeBatch) -> dict[str, Tensor]:
        """Apply one recurrent VDN update from padded whole episodes.

        ``batch.obs`` has shape ``(B, T+1, n_agents, obs_dim)``. Recurrence is
        preserved but detached at each eight-step trace boundary, matching the
        paper's truncated BPTT horizon.
        """
        if batch.obs.shape[2] != self.n_agents:
            raise ValueError("batch agent dimension does not match learner")
        online_q = self._sequence_q(self.q_network, batch.obs[:, :-1], grad=True)
        chosen = online_q.gather(-1, batch.actions.long().unsqueeze(-1)).squeeze(-1)
        q_total = chosen.sum(dim=-1)
        with torch.no_grad():
            target_q = self._sequence_q(self.target_q_network, batch.obs[:, 1:], grad=False)
            target_max_total = target_q.max(dim=-1).values.sum(dim=-1)
            targets = self._lambda_targets(
                batch.rewards, batch.dones, batch.mask, target_max_total,
            )
        valid = batch.mask.sum().clamp_min(1.0)
        loss = (F.smooth_l1_loss(q_total, targets, reduction="none") * batch.mask).sum() / valid
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.detach(), "q_total": q_total.detach(), "targets": targets.detach()}

    def _sequence_q(self, network: VDNDuelingLSTM, obs: Tensor, *, grad: bool) -> Tensor:
        inputs = self._append_roles(obs)
        batch_size, time_steps, n_agents, input_dim = inputs.shape
        hidden = None
        outputs = []
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            for start in range(0, time_steps, self.trace_length):
                end = min(start + self.trace_length, time_steps)
                segment = inputs[:, start:end].permute(0, 2, 1, 3).reshape(
                    batch_size * n_agents, end - start, input_dim
                )
                segment_q, hidden = network(segment, hidden)
                hidden = tuple(value.detach() for value in hidden)
                outputs.append(
                    segment_q.reshape(batch_size, n_agents, end - start, self.action_dim)
                    .permute(0, 2, 1, 3)
                )
        return torch.cat(outputs, dim=1)

    def _append_roles(self, obs: Tensor) -> Tensor:
        if not self.role_information:
            return obs
        roles = torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype)
        roles = roles.view(1, 1, self.n_agents, self.n_agents).expand(*obs.shape[:-1], self.n_agents)
        return torch.cat([obs, roles], dim=-1)

    def _lambda_targets(
        self, rewards: Tensor, dones: Tensor, mask: Tensor, next_values: Tensor,
    ) -> Tensor:
        targets = torch.zeros_like(rewards)
        time_steps = rewards.shape[1]
        for start in range(0, time_steps, self.trace_length):
            end = min(start + self.trace_length, time_steps)
            running = next_values[:, end - 1]
            for step in range(end - 1, start - 1, -1):
                bootstrap = next_values[:, step]
                if step < end - 1:
                    traced = (1.0 - self.trace_lambda) * bootstrap + self.trace_lambda * running
                    bootstrap = torch.where(mask[:, step + 1].bool(), traced, bootstrap)
                running = rewards[:, step] + self.gamma * (1.0 - dones[:, step]) * bootstrap
                running = running * mask[:, step]
                targets[:, step] = running
        return targets

    def maybe_update_targets(self, training_episode: int) -> bool:
        if training_episode <= 0 or training_episode % self.target_update_interval != 0:
            return False
        self.update_targets()
        return True

    def update_targets(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())


__all__ = ["VDNAgent", "VDNDuelingLSTM"]
