"""Connectivity-Driven Communication (CDC), end to end.

Paper: Pesce and Montana, "Learning Multi-Agent Coordination through
Connectivity-driven Communication," Machine Learning 112 (2023), arXiv:2002.05233.

Model: a shared pair encoder produces symmetric messages and graph weights; a matrix
heat kernel selects incoming messages; a shared message-only actor chooses actions; and
one recurrent centralized critic trains the complete communication policy.
Invariants: ``CDCPolicy`` follows Equations 1--7, while ``ReleasedCDCPolicy`` preserves
the archived program's directed pairs and elementwise exponential for reproduction.
Interface: ``CDCAgent`` owns acting, actor/critic updates, and both target networks.
Why: the complete published algorithm is kept here so its communication mechanism can
be followed without the experimental spectral and factorised variants.

Reference implementation: the authors' unreleased reference code (private Bitbucket
repository ``cdc``, revision ``46d287376c31cb183006b01542411d97cf95679a``, its
HEAD; not publicly accessible). Each divergence below was read from it directly: ``torch.exp`` is the
elementwise exponential rather than the paper's matrix exponential; ``HK[HK==0] =
HK_diff[HK==0]`` stores the relative change itself as the weight rather than the
heat-kernel value; and ``np.arange(0.05, 15, 0.05)`` is half-open, giving 299 grid points.
The released variant reproduces that program; the canonical variant restores the paper's
symmetric pairs, matrix exponential, Equation (5) value at ``p_hat``, and P=300.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..components import gumbel_policy_sample, soft_update_module


class ReleasedCDCEdgeNetwork(nn.Module):
    """The shared ordered-pair encoder from the released CDC implementation."""

    def __init__(self, obs_dim: int, message_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(2 * obs_dim)
        self.fc1 = nn.Linear(2 * obs_dim, message_dim)
        self.fc2 = nn.Linear(message_dim, message_dim)
        self.fc3 = nn.Linear(message_dim, 1)

    def forward(self, sender_obs: Tensor, receiver_obs: Tensor) -> tuple[Tensor, Tensor]:
        pair = self.input_norm(torch.cat([sender_obs, receiver_obs], dim=-1))
        message = self.fc2(F.relu(self.fc1(pair)))
        # Released code deliberately feeds the unrectified message to the edge head.
        weight = torch.sigmoid(self.fc3(message)).squeeze(-1)
        return message, weight


class ReleasedCDCActionNetwork(nn.Module):
    """Released message-only action head, including its per-agent BatchNorm calls."""

    def __init__(self, message_dim: int, action_dim: int) -> None:
        super().__init__()
        self.message_norm = nn.BatchNorm1d(message_dim)
        self.fc1 = nn.Linear(message_dim, message_dim)
        self.fc2 = nn.Linear(message_dim, message_dim)
        self.fc3 = nn.Linear(message_dim, action_dim)

    def forward(self, message: Tensor) -> Tensor:
        hidden = F.relu(self.fc1(self.message_norm(message)))
        hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden)


class ReleasedCDCPolicy(nn.Module):
    """CDC exactly as executed by the authors' 2020 Bitbucket release.

    The graph is directed and includes self edges.  The release applies ``exp``
    elementwise to the normalised Laplacian and uses the first relative change below
    0.05 as the pair weight; these differ from a conventional matrix heat kernel but
    are preserved here so the historical results can be reproduced.  Only the outer
    batched API and modern, non-in-place autograd expressions are adaptations.
    """

    def __init__(self, obs_dim: int, action_dim: int, message_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.edge_network = ReleasedCDCEdgeNetwork(obs_dim, message_dim)
        self.action_network = ReleasedCDCActionNetwork(message_dim, action_dim)
        self.register_buffer("diffusion_times", torch.arange(0.05, 15.0, 0.05))

    def compute_messages(self, obs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, n_agents, _ = obs.shape
        pair_messages = obs.new_empty(batch_size, n_agents, n_agents, self.message_dim)
        adjacency = obs.new_empty(batch_size, n_agents, n_agents)
        # Separate calls preserve the release's BatchNorm statistics per ordered pair.
        for sender in range(n_agents):
            for receiver in range(n_agents):
                message, weight = self.edge_network(obs[:, sender], obs[:, receiver])
                pair_messages[:, sender, receiver] = message
                adjacency[:, sender, receiver] = weight

        degree = adjacency.sum(dim=2)
        laplacian = torch.diag_embed(degree) - adjacency
        inv_sqrt = torch.diag_embed(degree.rsqrt())
        norm_laplacian = inv_sqrt @ laplacian @ inv_sqrt

        selected = self._diffusion_weights(norm_laplacian)

        weighted_pairs = selected.unsqueeze(-1) * pair_messages
        messages = weighted_pairs.sum(dim=1)
        return messages, {
            "adjacency": adjacency,
            "heat_weights": selected,
            "pair_messages": pair_messages,
            "messages": messages,
            "normalised_laplacian": norm_laplacian,
        }

    def _diffusion_weights(self, norm_laplacian: Tensor) -> Tensor:
        """Reproduce the release's elementwise exponential and stopping value."""
        heat = torch.exp(
            -self.diffusion_times.view(1, -1, 1, 1) * norm_laplacian.unsqueeze(1)
        )
        relative_change = ((heat[:, 1:] - heat[:, :-1]) / heat[:, :-1]).abs()
        stable = relative_change < 0.05
        has_stable = stable.any(dim=1)
        first_index = stable.to(dtype=torch.int64).argmax(dim=1)
        selected = torch.gather(relative_change, 1, first_index.unsqueeze(1)).squeeze(1)
        return torch.where(has_stable, selected, torch.zeros_like(selected))

    def forward(self, obs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        messages, diagnostics = self.compute_messages(obs)
        logits = torch.stack(
            [self.action_network(messages[:, agent]) for agent in range(messages.shape[1])],
            dim=1,
        )
        return logits, diagnostics

    def sample_gumbel(
        self, obs: Tensor, temperature: float = 1.0, hard: bool = True, deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        logits, diagnostics = self(obs)
        one_hot, indices, logits = gumbel_policy_sample(
            logits, action_dim=self.action_dim, temperature=temperature, hard=hard, deterministic=deterministic,
        )
        diagnostics["logits"] = logits
        diagnostics["action_one_hot"] = one_hot
        return one_hot, indices, diagnostics


class PaperEquationCDCPolicy(ReleasedCDCPolicy):
    """Paper-equation CDC with symmetric pairs and a matrix heat kernel.

    Equations 1--2 define ``c_uv = c_vu`` and ``s_uv = s_vu``.  We evaluate the
    released shared pair network in both directions and average the two outputs, which
    preserves its architecture while enforcing those equations.  Equation 4 uses the
    matrix exponential and Equation 5 retains the heat-kernel value at the first stable
    diffusion time.  Self pairs and the message-only action path follow Equations 1 and 7.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        message_dim: int = 64,
        *,
        diffusion_steps: int = 300,
        diffusion_max: float = 15.0,
        stable_delta: float = 0.05,
    ) -> None:
        super().__init__(obs_dim, action_dim, message_dim)
        # Section 4.2 specifies P=300 and s=0.05; the release's half-open
        # `np.arange(0.05, 15, 0.05)` yields only 299 points.
        self.diffusion_times = torch.linspace(0.05, diffusion_max, diffusion_steps)
        self.stable_delta = stable_delta

    def compute_messages(self, obs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        pair_messages, adjacency = self._symmetric_pairs(obs)
        degree = adjacency.sum(dim=2)
        laplacian = torch.diag_embed(degree) - adjacency
        inv_sqrt = torch.diag_embed(degree.clamp_min(1e-8).rsqrt())
        norm_laplacian = inv_sqrt @ laplacian @ inv_sqrt
        selected = self._diffusion_weights(norm_laplacian)
        messages = (selected.unsqueeze(-1) * pair_messages).sum(dim=1)
        return messages, {
            "adjacency": adjacency,
            "heat_weights": selected,
            "pair_messages": pair_messages,
            "messages": messages,
            "normalised_laplacian": norm_laplacian,
        }

    def _symmetric_pairs(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, n_agents, _ = obs.shape
        directed_messages = obs.new_empty(batch_size, n_agents, n_agents, self.message_dim)
        directed_adjacency = obs.new_empty(batch_size, n_agents, n_agents)
        for sender in range(n_agents):
            for receiver in range(n_agents):
                message, weight = self.edge_network(obs[:, sender], obs[:, receiver])
                directed_messages[:, sender, receiver] = message
                directed_adjacency[:, sender, receiver] = weight
        pair_messages = 0.5 * (directed_messages + directed_messages.transpose(1, 2))
        adjacency = 0.5 * (directed_adjacency + directed_adjacency.transpose(1, 2))
        return pair_messages, adjacency

    def _diffusion_weights(self, norm_laplacian: Tensor) -> Tensor:
        heat = torch.matrix_exp(
            -self.diffusion_times.view(1, -1, 1, 1) * norm_laplacian.unsqueeze(1)
        )
        previous, current = heat[:, :-1], heat[:, 1:]
        relative_change = ((current - previous) / previous.clamp_min(1e-8)).abs()
        stable = relative_change < self.stable_delta
        has_stable = stable.any(dim=1)
        first_index = stable.to(dtype=torch.int64).argmax(dim=1)
        # Equation (5) tests (H(p+1) - H(p)) / H(p) < delta and then evaluates the kernel at
        # p_hat = p, the point the test was anchored on -- so the value comes from
        # `previous`, not from the next grid point. The release takes a different route
        # entirely and stores the relative change itself as the weight; see ReleasedCDCPolicy.
        selected = torch.gather(previous, 1, first_index.unsqueeze(1)).squeeze(1)
        return torch.where(has_stable, selected, torch.zeros_like(selected))


class CDCPolicy(PaperEquationCDCPolicy):
    """Canonical CDC implementing the paper's symmetric heat-diffusion equations."""


@dataclass(frozen=True)
class CDCUpdate:
    """Scalar diagnostics from one complete CDC actor-critic update."""

    critic_loss: float
    actor_loss: float
    target_q: float
    mean_edge_weight: float


@dataclass(frozen=True)
class CDCConfig:
    """Training schedule and hyperparameters reported for Navigation Control."""

    batch_size: int = 1024
    episode_horizon: int = 25
    update_interval: int = 100
    replay_capacity: int = 1_000_000

    def update_due(self, *, environment_steps: int, replay_size: int) -> bool:
        return replay_size >= self.batch_size and environment_steps % self.update_interval == 0


class ReleasedCDCCritic(nn.Module):
    """The released critic: shared pair encoder, agent-order LSTMCell, linear Q head."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.recurrent = nn.LSTMCell(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor, action_one_hot: Tensor) -> Tensor:
        batch_size = obs.shape[0]
        hidden = obs.new_zeros(batch_size, self.recurrent.hidden_size)
        cell = torch.zeros_like(hidden)
        for agent in range(obs.shape[1]):
            encoded = F.relu(self.encoder(torch.cat([obs[:, agent], action_one_hot[:, agent]], dim=-1)))
            hidden, cell = self.recurrent(encoded, (hidden, cell))
        return self.q_head(hidden).squeeze(-1)


class CDCAgent(nn.Module):
    """Complete CDC learner: communication actor, recurrent critic, and targets.

    ``variant='paper'`` implements the published symmetric matrix-heat equations;
    ``variant='released'`` reproduces the archived executable implementation.  Both use
    the paper's shared centralized LSTM critic and off-policy Gumbel actor update.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        message_dim: int = 64,
        *,
        variant: str = "paper",
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        policy_regularization: float = 1e-3,
        max_grad_norm: float = 0.5,
        diffusion_steps: int = 300,
        diffusion_max: float = 15.0,
        stable_delta: float = 0.05,
    ) -> None:
        super().__init__()
        if variant not in {"paper", "released"}:
            raise ValueError("variant must be 'paper' or 'released'")
        policy_type = CDCPolicy if variant == "paper" else ReleasedCDCPolicy
        # Equation (5)'s search grid is configurable only on the paper variant; the released
        # one is pinned to its own `np.arange(0.05, 15, 0.05)`.
        grid = (
            {"diffusion_steps": diffusion_steps, "diffusion_max": diffusion_max,
             "stable_delta": stable_delta}
            if variant == "paper" else {}
        )
        self.action_dim = action_dim
        self.variant = variant
        self.gamma = gamma
        self.tau = tau
        self.policy_regularization = policy_regularization
        self.max_grad_norm = max_grad_norm
        self.actor = policy_type(obs_dim, action_dim, message_dim, **grid)
        self.critic = ReleasedCDCCritic(obs_dim, action_dim, hidden_dim)
        # The release constructs fresh targets before the hard copy. Besides matching its
        # object lifecycle, this preserves the published seeded random-number stream.
        self.target_actor = policy_type(obs_dim, action_dim, message_dim, **grid)
        self.target_critic = ReleasedCDCCritic(obs_dim, action_dim, hidden_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    @torch.no_grad()
    def act(self, obs: Tensor, *, deterministic: bool = False) -> Tensor:
        """Return integer joint actions for observations ``(batch, agents, obs_dim)``."""
        training = self.actor.training
        self.actor.eval()
        _, actions, _ = self.actor.sample_gumbel(obs, hard=True, deterministic=deterministic)
        self.actor.train(training)
        return actions

    @torch.no_grad()
    def act_without_messages(self, obs: Tensor) -> Tensor:
        """Ablate CDC by feeding the shared action head an all-zero message."""
        training = self.actor.training
        self.actor.eval()
        zero_message = obs.new_zeros(obs.shape[0], self.actor.message_dim)
        logits = torch.stack(
            [self.actor.action_network(zero_message) for _ in range(obs.shape[1])], dim=1,
        )
        self.actor.train(training)
        return logits.argmax(dim=-1)

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> CDCUpdate:
        """Apply the released centralized-Q update to one normalized-reward batch."""
        replay_actions = F.one_hot(actions.long(), self.action_dim).float()
        with torch.no_grad():
            next_actions, _, _ = self.target_actor.sample_gumbel(
                next_obs, hard=True, deterministic=True,
            )
            target_q = rewards + self.gamma * (1.0 - dones) * self.target_critic(
                next_obs, next_actions,
            )
        critic_loss = F.mse_loss(self.critic(obs, replay_actions), target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        policy_actions, _, diagnostics = self.actor.sample_gumbel(obs, hard=True)
        actor_loss = -self.critic(obs, policy_actions).mean()
        actor_loss = actor_loss + self.policy_regularization * (
            diagnostics["logits"].square().mean(dim=(0, 2)).sum()
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        self.soft_update()
        return CDCUpdate(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            target_q=float(target_q.mean()),
            mean_edge_weight=float(diagnostics["adjacency"].mean().detach()),
        )

    @torch.no_grad()
    def soft_update(self) -> None:
        soft_update_module(self.target_actor, self.actor, self.tau)
        soft_update_module(self.target_critic, self.critic, self.tau)


__all__ = [
    "CDCAgent",
    "CDCConfig",
    "CDCPolicy",
    "CDCUpdate",
    "PaperEquationCDCPolicy",
    "ReleasedCDCCritic",
    "ReleasedCDCPolicy",
]
