"""NDQ implementation.

Original paper:
Tonghan Wang, Jianhao Wang, Chongyi Zheng, Chongjie Zhang. "Learning Nearly
Decomposable Value Functions Via Communication Minimization." International
Conference on Learning Representations (ICLR), 2020. arXiv:1910.05366.
Official code: https://github.com/TonghanWang/NDQ (Apache-2.0), revision
575f2e243bac1a567c072dbea8e093aaa4959511, used as the mechanism reference;
adapted onto modMARL's episode-replay scaffold.

QMIX with a learned pairwise message channel. Each agent's observation is encoded
into a unit-variance Gaussian message per teammate, m_ij ~ N(f_m(o_i, j), I); the
reparameterised samples are concatenated (sender-ordered) into the recipient's
recurrent Q-network input, and per-agent chosen-action Qs mix into Q_tot exactly as
in QMIX. Two information-theoretic losses shape the channel (paper Eq. 7): an
expressiveness term — a shared variational posterior q_xi must predict the
recipient's greedy action from its observation and incoming messages — and a
succinctness term — KL(N(mu, I) || N(0, I)) = mu^2/2, pulling useless message bits
to zero so they can be cut at execution time by mean magnitude with little loss.

Deviations from the official code, recorded deliberately: (1) the self-message
chunk m_jj is zeroed (the paper's semantics; the official code leaves it in the
recipient's input); (2) the code-only third entropy loss term is dropped (absent
from the paper and set to 0 in the README reference runs); (3) execution-time
cutting supports the absolute |mu| threshold only (no rank mode); and (4)
auxiliary losses are masked elementwise over padding (the official divides by
mask.sum() without zeroing padded entries). The shared controller inputs, RMSProp
optimizer, recurrent learner, and hard target schedule match the release.

The paper's main StarCraft experiments use lambda=0.1 and beta=1e-5.  The released
repository is inconsistent: its base YAML uses 1e-3 while README commands use 1e-2
for hallway and 1e-4 for StarCraft.  The defaults below follow the paper; callers may
select a documented task-specific setting explicitly.  The PDF appendix describes a
one-layer encoder and 20-unit posterior, while the released full-broadcast module uses
two 64-unit encoder layers and posterior widths ``4*n_agents*message_dim``; this port
follows that executable released variant.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.nn import build_mlp
from .qmix import QMixer


@dataclass(frozen=True)
class NDQUpdate:
    """Scalar diagnostics from one complete recurrent NDQ learner update."""

    loss: float
    td_loss: float
    expressiveness_loss: float
    succinctness_loss: float


class NDQMessageEncoder(nn.Module):
    """The message head f_m and the shared variational posterior q_xi."""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, message_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.mean_head = build_mlp(obs_dim, [hidden_dim, hidden_dim], n_agents * message_dim)
        posterior_hidden = 4 * n_agents * message_dim
        self.posterior = build_mlp(
            obs_dim + n_agents * message_dim, [posterior_hidden, posterior_hidden], action_dim,
        )

    def means(self, obs: Tensor) -> Tensor:
        # obs (B, n, obs_dim) -> mu (B, n_senders, n_receivers, message_dim)
        batch, n_agents, _ = obs.shape
        return self.mean_head(obs).view(batch, n_agents, n_agents, self.message_dim)


class _RecurrentQNetwork(nn.Module):
    """PyMARL-shape GRU Q head: Linear -> ReLU -> GRUCell -> Linear, shared across agents."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, inputs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch, n_agents, _ = inputs.shape
        x = torch.relu(self.fc1(inputs)).reshape(batch * n_agents, self.hidden_dim)
        new_hidden = self.gru(x, hidden.reshape(batch * n_agents, self.hidden_dim))
        new_hidden = new_hidden.view(batch, n_agents, self.hidden_dim)
        return self.head(new_hidden), new_hidden


class NDQAgent(nn.Module):
    """NDQ: message encoder + recurrent Q-network + QMIX mixer, with target copies."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        message_dim: int = 3,
        mixer_hidden_dim: int = 32,
        *,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        communication_weight: float = 0.1,
        succinctness_weight: float = 1e-5,
        max_grad_norm: float = 10.0,
        include_agent_id: bool = True,
        include_last_action: bool = True,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.communication_weight = communication_weight
        self.succinctness_weight = succinctness_weight
        self.max_grad_norm = max_grad_norm
        self.action_dim = action_dim
        self.include_agent_id = include_agent_id
        self.include_last_action = include_last_action
        controller_input_dim = obs_dim
        if include_last_action:
            controller_input_dim += action_dim
        if include_agent_id:
            controller_input_dim += n_agents
        self.message_encoder = NDQMessageEncoder(
            controller_input_dim, action_dim, n_agents, message_dim, hidden_dim,
        )
        self.q_network = _RecurrentQNetwork(
            controller_input_dim + n_agents * message_dim, action_dim, hidden_dim,
        )
        self.mixer = QMixer(n_agents, n_agents * obs_dim, mixer_hidden_dim)
        self.target_message_encoder = copy.deepcopy(self.message_encoder)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_message_encoder.requires_grad_(False)
        self.target_q_network.requires_grad_(False)
        self.target_mixer.requires_grad_(False)
        self.optimizer = torch.optim.RMSprop(
            list(self.message_encoder.parameters())
            + list(self.q_network.parameters())
            + list(self.mixer.parameters()),
            lr=learning_rate, alpha=0.99, eps=1e-5,
        )

    def init_hidden(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.n_agents, self.hidden_dim, device=device)

    def messages(
        self,
        obs: Tensor,
        last_actions: Tensor | None = None,
        *,
        target: bool = False,
        drop_threshold: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample the pairwise messages and route them receiver-major.

        obs (B, n, obs_dim) -> (m_in (B, n, n * message_dim), mu (B, n_send, n_recv, dim)).
        Reparameterised sample m = mu + eps keeps gradients flowing into the encoder;
        self-chunks are zeroed; with ``drop_threshold`` set, sampled bits whose |mu| is
        below the threshold are cut (zero-filled at the recipient), per the paper's
        execution-time protocol.
        """
        encoder = self.target_message_encoder if target else self.message_encoder
        controller_inputs = self._controller_inputs(obs, last_actions)
        mu = encoder.means(controller_inputs)                         # (B, send, recv, dim)
        sample = mu + torch.randn_like(mu)
        if drop_threshold is not None:
            sample = sample * (mu.abs() >= drop_threshold).to(sample.dtype)
        no_self = 1.0 - torch.eye(self.n_agents, device=obs.device, dtype=sample.dtype).view(
            1, self.n_agents, self.n_agents, 1,
        )
        sample = sample * no_self
        batch = obs.shape[0]
        m_in = sample.permute(0, 2, 1, 3).reshape(batch, self.n_agents, self.n_agents * self.message_dim)
        return m_in, mu

    def q_step(
        self,
        obs: Tensor,
        m_in: Tensor,
        hidden: Tensor,
        last_actions: Tensor | None = None,
        *,
        target: bool = False,
    ) -> tuple[Tensor, Tensor]:
        network = self.target_q_network if target else self.q_network
        return network(torch.cat([self._controller_inputs(obs, last_actions), m_in], dim=-1), hidden)

    def posterior_logits(
        self, obs: Tensor, m_in: Tensor, last_actions: Tensor | None = None,
    ) -> Tensor:
        inputs = self._controller_inputs(obs, last_actions)
        return self.message_encoder.posterior(torch.cat([inputs, m_in], dim=-1))

    def _controller_inputs(self, obs: Tensor, last_actions: Tensor | None) -> Tensor:
        """Build the released shared-controller input for every agent."""
        pieces = [obs]
        if self.include_last_action:
            if last_actions is None:
                last_actions = torch.zeros(
                    *obs.shape[:2], self.action_dim, device=obs.device, dtype=obs.dtype,
                )
            pieces.append(last_actions.to(dtype=obs.dtype))
        if self.include_agent_id:
            identity = torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype)
            pieces.append(identity.unsqueeze(0).expand(obs.shape[0], -1, -1))
        return torch.cat(pieces, dim=-1)

    def succinctness_loss(self, mu: Tensor, mask: Tensor) -> Tensor:
        """Paper Eq. 7 KL term, excluding nonexistent self-messages.

        ``mu`` has shape ``(B, T, sender, receiver, message_dim)`` and ``mask``
        has shape ``(B, T)``.  The diagonal is excluded before reducing because
        those self-message samples are never routed into an agent's Q-network.
        """
        no_self = 1.0 - torch.eye(self.n_agents, device=mu.device, dtype=mu.dtype)
        pairwise_kl = 0.5 * mu.square() * no_self.view(1, 1, self.n_agents, self.n_agents, 1)
        per_step = pairwise_kl.sum(dim=(-1, -2, -3)) * mask
        return per_step.sum() / mask.sum().clamp_min(1.0)

    def update(self, batch) -> NDQUpdate:
        """Apply one full recurrent double-Q/QMIX update with paper Eq. 7."""
        batch_size, horizon = batch.actions.shape[:2]
        n, device = self.n_agents, batch.obs.device

        hidden = self.init_hidden(batch_size, device)
        target_hidden = self.init_hidden(batch_size, device)
        online_q, target_q, online_mu, online_messages = [], [], [], []
        for timestep in range(horizon + 1):
            obs_t = batch.obs[:, timestep]
            last_actions = self._last_action_one_hot(batch.actions, timestep)
            messages, mu = self.messages(obs_t, last_actions)
            q_values, hidden = self.q_step(obs_t, messages, hidden, last_actions)
            online_q.append(q_values)
            online_mu.append(mu)
            online_messages.append(messages)
            with torch.no_grad():
                target_messages, _ = self.messages(obs_t, last_actions, target=True)
                target_values, target_hidden = self.q_step(
                    obs_t, target_messages, target_hidden, last_actions, target=True,
                )
                target_q.append(target_values)
        online_q_tensor = torch.stack(online_q, dim=1)
        target_q_tensor = torch.stack(target_q, dim=1)

        chosen = online_q_tensor[:, :-1].gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        state = batch.obs.reshape(batch_size, horizon + 1, -1)
        q_total = self.mixer(chosen.reshape(-1, n), state[:, :-1].reshape(-1, state.shape[-1]))
        q_total = q_total.view(batch_size, horizon)
        with torch.no_grad():
            next_actions = online_q_tensor[:, 1:].argmax(dim=-1, keepdim=True)
            next_target = target_q_tensor[:, 1:].gather(-1, next_actions).squeeze(-1)
            next_total = self.target_mixer(
                next_target.reshape(-1, n), state[:, 1:].reshape(-1, state.shape[-1]),
            ).view(batch_size, horizon)
            target = batch.rewards + self.gamma * (1.0 - batch.dones) * next_total
        td_error = (q_total - target) * batch.mask
        td_loss = td_error.square().sum() / batch.mask.sum().clamp_min(1.0)

        posterior_logits = torch.stack(
            [
                self.posterior_logits(
                    batch.obs[:, t], online_messages[t], self._last_action_one_hot(batch.actions, t),
                )
                for t in range(horizon)
            ],
            dim=1,
        )
        labels = target_q_tensor[:, :-1].argmax(dim=-1)
        cross_entropy = F.cross_entropy(
            posterior_logits.reshape(-1, posterior_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view(batch_size, horizon, n)
        expressiveness = (cross_entropy * batch.mask.unsqueeze(-1)).sum()
        expressiveness = expressiveness / batch.mask.sum().clamp_min(1.0)

        mu = torch.stack(online_mu[:-1], dim=1)
        succinctness = self.succinctness_loss(mu, batch.mask)
        loss = td_loss + self.communication_weight * (
            expressiveness + self.succinctness_weight * succinctness
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.message_encoder.parameters())
            + list(self.q_network.parameters())
            + list(self.mixer.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()
        return NDQUpdate(
            loss=float(loss.detach()),
            td_loss=float(td_loss.detach()),
            expressiveness_loss=float(expressiveness.detach()),
            succinctness_loss=float(succinctness.detach()),
        )

    def _last_action_one_hot(self, actions: Tensor, timestep: int) -> Tensor | None:
        if not self.include_last_action:
            return None
        if timestep == 0:
            return torch.zeros(
                actions.shape[0], self.n_agents, self.action_dim,
                device=actions.device, dtype=torch.float32,
            )
        return F.one_hot(actions[:, timestep - 1], num_classes=self.action_dim).to(torch.float32)

    def update_targets(self) -> None:
        """Hard-copy the online networks into the target networks."""
        self.target_message_encoder.load_state_dict(self.message_encoder.state_dict())
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def update_targets_if_due(self, episode: int, interval: int) -> bool:
        """Apply the released episode-count target schedule and report whether it fired."""
        if episode <= 0 or episode % interval != 0:
            return False
        self.update_targets()
        return True


__all__ = ["NDQAgent", "NDQMessageEncoder", "NDQUpdate"]
