"""QMIX from Rashid et al. (ICML 2018) and PyMARL alpha ``eaf6b06``.

Model: a parameter-shared GRU utility network consumes each local observation,
previous action, and agent identity. A state-conditioned monotonic mixer combines
selected utilities during centralized training; decentralized execution uses only
individual utilities. Episode replay, double Q-learning, action-availability
masking, RMSProp, gradient clipping, and hard target copies follow the release.

Adaptation: modMARL environments expose centralized task state through the runner.
When an external environment lacks that API, its adapter must construct a factual
global state; concatenated observations are not silently treated as state here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class QMIXBatch:
    obs: Tensor                 # (B, T+1, N, obs_dim)
    states: Tensor              # (B, T+1, state_dim)
    actions: Tensor             # (B, T, N)
    available_actions: Tensor   # (B, T+1, N, action_dim)
    rewards: Tensor             # (B, T)
    dones: Tensor               # (B, T)
    mask: Tensor                # (B, T)


class QMIXReplayBuffer:
    """Whole-episode storage for QMIX's state and availability-dependent update.

    Local observations and centralized state are deliberately stored separately:
    they are different signals in QMIX, even when an adapter derives both from the
    same underlying environment snapshot.
    """

    def __init__(
        self, capacity: int, horizon: int, n_agents: int, obs_dim: int,
        state_dim: int, action_dim: int,
    ) -> None:
        self.capacity, self.horizon = capacity, horizon
        self.obs = np.zeros((capacity, horizon + 1, n_agents, obs_dim), np.float32)
        self.states = np.zeros((capacity, horizon + 1, state_dim), np.float32)
        self.actions = np.zeros((capacity, horizon, n_agents), np.int64)
        self.available = np.zeros((capacity, horizon + 1, n_agents, action_dim), np.float32)
        self.rewards = np.zeros((capacity, horizon), np.float32)
        self.dones = np.zeros((capacity, horizon), np.float32)
        self.mask = np.zeros((capacity, horizon), np.float32)
        self.size = self.ptr = 0

    def add_episode(
        self, *, obs: np.ndarray, states: np.ndarray, actions: np.ndarray,
        available_actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray,
    ) -> None:
        length = len(actions)
        if length > self.horizon:
            raise ValueError("episode exceeds replay horizon")
        for array in (self.obs, self.states, self.actions, self.available,
                      self.rewards, self.dones, self.mask):
            array[self.ptr] = 0
        self.obs[self.ptr, :length + 1] = obs
        self.states[self.ptr, :length + 1] = states
        self.actions[self.ptr, :length] = actions
        self.available[self.ptr, :length + 1] = available_actions
        self.rewards[self.ptr, :length] = rewards
        self.dones[self.ptr, :length] = dones
        self.mask[self.ptr, :length] = 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> QMIXBatch:
        indices = np.random.randint(0, self.size, batch_size)
        def tensor(array: np.ndarray) -> Tensor:
            return torch.as_tensor(array[indices], device=device)
        return QMIXBatch(
            tensor(self.obs), tensor(self.states), tensor(self.actions),
            tensor(self.available), tensor(self.rewards), tensor(self.dones), tensor(self.mask),
        )

    def __len__(self) -> int:
        return self.size


class AgentQNetwork(nn.Module):
    """PyMARL DRQN utility: linear-ReLU, 64-unit GRUCell, linear action head."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inputs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        encoded = F.relu(self.fc1(inputs))
        hidden = self.gru(encoded, hidden)
        return self.fc2(hidden), hidden


class QMixer(nn.Module):
    """Paper's state-conditioned two-layer monotonic mixing network."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixer_hidden_dim: int = 32,
        hypernet_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_agents, self.state_dim, self.mixer_hidden_dim = n_agents, state_dim, mixer_hidden_dim
        def weight_hypernet(output_dim: int) -> nn.Module:
            if hypernet_hidden_dim is None:
                return nn.Linear(state_dim, output_dim)
            return nn.Sequential(
                nn.Linear(state_dim, hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(hypernet_hidden_dim, output_dim),
            )

        self.hyper_w1 = weight_hypernet(n_agents * mixer_hidden_dim)
        self.hyper_w2 = weight_hypernet(mixer_hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixer_hidden_dim)
        self.state_value = nn.Sequential(
            nn.Linear(state_dim, mixer_hidden_dim), nn.ReLU(), nn.Linear(mixer_hidden_dim, 1)
        )

    def forward(self, agent_qs: Tensor, state: Tensor) -> Tensor:
        leading = agent_qs.shape[:-1]
        flat_q = agent_qs.reshape(-1, 1, self.n_agents)
        flat_state = state.reshape(-1, self.state_dim)
        w1 = self.hyper_w1(flat_state).abs().view(-1, self.n_agents, self.mixer_hidden_dim)
        b1 = self.hyper_b1(flat_state).view(-1, 1, self.mixer_hidden_dim)
        hidden = F.elu(torch.bmm(flat_q, w1) + b1)
        w2 = self.hyper_w2(flat_state).abs().view(-1, self.mixer_hidden_dim, 1)
        value = self.state_value(flat_state).view(-1, 1, 1)
        return (torch.bmm(hidden, w2) + value).view(*leading)


class QMIXAgent(nn.Module):
    """Complete PyMARL-alpha QMIX learner and decentralized recurrent controller."""

    def __init__(
        self, n_agents: int, obs_dim: int, action_dim: int, state_dim: int | None = None,
        hidden_dim: int = 64, mixer_hidden_dim: int = 32, *, learning_rate: float = 5e-4,
        gamma: float = 0.99, target_update_interval: int = 200,
        epsilon_start: float = 1.0, epsilon_end: float = 0.05,
        epsilon_anneal_steps: int = 20_000,
    ) -> None:
        super().__init__()
        self.n_agents, self.obs_dim, self.action_dim = n_agents, obs_dim, action_dim
        self.state_dim = state_dim if state_dim is not None else n_agents * obs_dim
        self.gamma, self.target_update_interval = gamma, target_update_interval
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_anneal_steps = epsilon_anneal_steps
        input_dim = obs_dim + action_dim + n_agents
        self.q_network = AgentQNetwork(input_dim, action_dim, hidden_dim)
        self.mixer = QMixer(n_agents, self.state_dim, mixer_hidden_dim)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.optimizer = torch.optim.RMSprop(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=learning_rate, alpha=0.99, eps=1e-5,
        )
        self._execution_hidden: Tensor | None = None
        self._last_actions: Tensor | None = None

    def epsilon(self, env_step: int) -> float:
        fraction = min(max(env_step, 0) / self.epsilon_anneal_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def reset_hidden(self, device: torch.device | None = None) -> None:
        device = device or next(self.parameters()).device
        self._execution_hidden = torch.zeros(
            self.n_agents, self.q_network.hidden_dim, device=device,
        )
        self._last_actions = torch.zeros(self.n_agents, self.action_dim, device=device)

    @torch.no_grad()
    def act(self, obs: Tensor, available_actions: Tensor, epsilon: float) -> Tensor:
        if self._execution_hidden is None:
            self.reset_hidden(obs.device)
        roles = torch.eye(self.n_agents, device=obs.device)
        inputs = torch.cat([obs, self._last_actions, roles], dim=-1)
        q_values, self._execution_hidden = self.q_network(inputs, self._execution_hidden)
        masked_q = q_values.masked_fill(available_actions == 0, -torch.inf)
        greedy = masked_q.argmax(dim=-1)
        random_scores = torch.rand_like(q_values).masked_fill(available_actions == 0, -1.0)
        random_actions = random_scores.argmax(dim=-1)
        actions = torch.where(torch.rand(self.n_agents, device=obs.device) < epsilon,
                              random_actions, greedy)
        self._last_actions = F.one_hot(actions, self.action_dim).to(obs.dtype)
        return actions

    def update(self, batch: QMIXBatch) -> dict[str, Tensor]:
        online = self._unroll(self.q_network, batch.obs, batch.actions)
        chosen = online[:, :-1].gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        q_total = self.mixer(chosen, batch.states[:, :-1])
        with torch.no_grad():
            target = self._unroll(self.target_q_network, batch.obs, batch.actions)[:, 1:]
            online_next = online[:, 1:].masked_fill(batch.available_actions[:, 1:] == 0, -torch.inf)
            greedy_next = online_next.argmax(dim=-1, keepdim=True)
            target = target.masked_fill(batch.available_actions[:, 1:] == 0, -torch.inf)
            target_utilities = target.gather(-1, greedy_next).squeeze(-1)
            target_total = self.target_mixer(target_utilities, batch.states[:, 1:])
            td_target = batch.rewards + self.gamma * (1.0 - batch.dones) * target_total
        mask = batch.mask.clone()
        mask[:, 1:] *= 1.0 - batch.dones[:, :-1]
        td_error = (q_total - td_target) * mask
        loss = td_error.square().sum() / mask.sum().clamp_min(1.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixer.parameters()), 10.0,
        )
        self.optimizer.step()
        return {"loss": loss.detach(), "grad_norm": grad_norm.detach(),
                "q_total": q_total.detach(), "targets": td_target.detach()}

    def _unroll(self, network: AgentQNetwork, obs: Tensor, actions: Tensor) -> Tensor:
        batch_size, time_steps, _, _ = obs.shape
        hidden = torch.zeros(
            batch_size * self.n_agents, network.hidden_dim, device=obs.device,
        )
        previous = torch.zeros(
            batch_size, self.n_agents, self.action_dim, device=obs.device,
        )
        roles = torch.eye(self.n_agents, device=obs.device).view(1, self.n_agents, self.n_agents)
        roles = roles.expand(batch_size, -1, -1)
        outputs = []
        for step in range(time_steps):
            inputs = torch.cat([obs[:, step], previous, roles], dim=-1).reshape(
                batch_size * self.n_agents, -1
            )
            q_values, hidden = network(inputs, hidden)
            outputs.append(q_values.view(batch_size, self.n_agents, self.action_dim))
            if step < actions.shape[1]:
                previous = F.one_hot(actions[:, step].long(), self.action_dim).float()
        return torch.stack(outputs, dim=1)

    def maybe_update_targets(self, episode: int) -> bool:
        if episode <= 0 or episode % self.target_update_interval != 0:
            return False
        self.update_targets()
        return True

    def update_targets(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())


__all__ = ["AgentQNetwork", "QMIXAgent", "QMIXBatch", "QMIXReplayBuffer", "QMixer"]
