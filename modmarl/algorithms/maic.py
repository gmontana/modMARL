"""Paper-faithful Multi-Agent Incentive Communication (MAIC).

Model: a shared 64-unit GRU consumes each agent's observation, previous
one-hot action, and identity.  Its state parameterises one diagonal Gaussian per modeled
teammate; sender state and sampled teammate latent produce action-value incentives.
Invariants: Gaussian output stores all means before all variances, directed self
pairs never contribute to Equation 1, and receiver values sum sender incentives.
Interface: ``MAICAgent.step`` performs one recurrent policy step and the two loss
methods implement paper Equations 1 and 4.  VDN and QMIX are explicit mixers.
Why: this follows Yuan et al. (AAAI 2022) and
``mansicer/MAIC@2bd47d105ccd64bfba1f1d71981f7723c59ac07f``.  Three release bugs are
corrected in favour of the paper, each verified against the release: the sender pairing
(the release's ``h_repeat`` yields the *receiver's* history where paper Equation 1 wants
the sender's), the unmasked entropy recomputation in Equation 4, and the supervision of
Equation 1 -- the release regresses onto the current greedy action of the
message-augmented Q, while the paper's Theorem 1 states the action is taken from the
replay buffer, which is what ``teammate_model_loss`` uses.
The previous action *is* a policy input: ``join1.yaml`` sets ``obs_last_action: False``
only inside ``env_args``, where ``Join1Env`` stores it and never reads it, so the
controller sees the top-level ``default.yaml`` value ``True`` -- and the paper agrees
("observation and last action").  The paper reports VDN
for Hallway while the pinned config selects QMIX; both remain explicit. Hallway uses
three agents, lengths (2, 6, 10), horizon 20, and simultaneous-arrival reward.

Normalization is another release detail: ``test_mode=True`` uses mean latents
and pruned attention but never switches BatchNorm to running statistics. The
training example reproduces this batch-statistics evaluation and crops replay to
the longest filled episode as the official runner does. Evaluation statistics
span the current team's agents; this is not strictly local message generation.
Direct ``agent.eval()`` still has ordinary PyTorch semantics. See
``tools/reference/check_maic.py`` for a pinned encoder-normalization comparison;
it does not assert whole-policy or learning-curve parity.
"""

from __future__ import annotations

import copy
import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from .qmix import QMixer


class MAICStep(NamedTuple):
    q: Tensor
    q_loc: Tensor
    mu: Tensor
    sigma: Tensor
    z: Tensor
    alpha: Tensor
    hidden: Tensor


class _LeadingBatchMLP(nn.Module):
    """Released Linear-BatchNorm-LeakyReLU-Linear MLP on leading dimensions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, batch_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend((nn.LeakyReLU(), nn.Linear(hidden_dim, output_dim)))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape[:-1]
        return self.layers(inputs.reshape(-1, inputs.shape[-1])).view(*shape, -1)


class _MAICNetwork(nn.Module):
    """Shared recurrent policy, teammate posterior, and incentive channel."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int,
                 latent_dim: int, attention_dim: int, var_floor: float,
                 include_previous_action: bool) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.attention_dim = attention_dim
        self.var_floor = var_floor
        self.include_previous_action = include_previous_action
        policy_input_dim = obs_dim + n_agents + (action_dim if include_previous_action else 0)
        self.fc1 = nn.Linear(policy_input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)
        self.embed_net = _LeadingBatchMLP(hidden_dim, 64, 2 * n_agents * latent_dim, batch_norm=True)
        self.inference_net = _LeadingBatchMLP(hidden_dim + action_dim, 64, 2 * latent_dim, batch_norm=True)
        self.msg_net = _LeadingBatchMLP(hidden_dim + latent_dim, 64, action_dim, batch_norm=False)
        self.w_query = nn.Linear(hidden_dim, attention_dim)
        self.w_key = nn.Linear(latent_dim, attention_dim)

    def _gaussian(self, stats: Tensor, *, modeled_agents: bool = False) -> tuple[Tensor, Tensor]:
        split = self.n_agents * self.latent_dim if modeled_agents else self.latent_dim
        means, raw_variances = stats.split(split, dim=-1)
        variance = raw_variances.exp().clamp_min(self.var_floor)
        if modeled_agents:
            means = means.view(*means.shape[:-1], self.n_agents, self.latent_dim)
            variance = variance.view(*variance.shape[:-1], self.n_agents, self.latent_dim)
        return means, variance.sqrt()

    def attention(self, hidden: Tensor, z: Tensor) -> Tensor:
        """Equation 2: self-masked scaled weights over receivers per sender."""
        query = self.w_query(hidden)
        key = self.w_key(z)
        logits = (query.unsqueeze(2) * key).sum(-1) / math.sqrt(self.attention_dim)
        self_mask = torch.eye(self.n_agents, device=logits.device, dtype=torch.bool).unsqueeze(0)
        return torch.softmax(logits.masked_fill(self_mask, -torch.inf), dim=-1)

    def forward(self, obs: Tensor, previous_actions: Tensor, hidden: Tensor, *,
                deterministic: bool = False, prune_threshold: float | None = None) -> MAICStep:
        batch, n, _ = obs.shape
        identities = torch.eye(n, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(batch, -1, -1)
        pieces = (obs, previous_actions, identities) if self.include_previous_action else (obs, identities)
        inputs = torch.cat(pieces, dim=-1)
        encoded = torch.relu(self.fc1(inputs)).reshape(batch * n, self.hidden_dim)
        new_hidden = self.gru(encoded, hidden.reshape(batch * n, self.hidden_dim)).view(batch, n, -1)
        q_loc = self.q_head(new_hidden)

        mu, sigma = self._gaussian(self.embed_net(new_hidden), modeled_agents=True)
        z = mu if deterministic else mu + sigma * torch.randn_like(sigma)
        sender_hidden = new_hidden.unsqueeze(2).expand(-1, -1, n, -1)
        messages = self.msg_net(torch.cat((sender_hidden, z), dim=-1))
        alpha = self.attention(new_hidden, z)
        if prune_threshold is not None:
            alpha = alpha * (alpha >= prune_threshold / n).to(alpha.dtype)
        q = q_loc + (alpha.unsqueeze(-1) * messages).sum(dim=1)
        return MAICStep(q, q_loc, mu, sigma, z, alpha, new_hidden)


class MAICAgent(nn.Module):
    """MAIC policy with target networks and an explicit VDN/QMIX mixer."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 latent_dim: int = 8, attention_dim: int = 32, mixer_hidden_dim: int = 32,
                 var_floor: float = 0.002, mixer: Literal["vdn", "qmix"] = "qmix",
                 include_previous_action: bool = True) -> None:
        super().__init__()
        if mixer not in {"vdn", "qmix"}:
            raise ValueError("mixer must be 'vdn' or 'qmix'")
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mixer_type = mixer
        self.network = _MAICNetwork(
            n_agents, obs_dim, action_dim, hidden_dim, latent_dim, attention_dim,
            var_floor, include_previous_action,
        )
        self.mixer = (
            QMixer(
                n_agents,
                n_agents * obs_dim,
                mixer_hidden_dim,
                hypernet_hidden_dim=64,
            )
            if mixer == "qmix"
            else None
        )
        self.target_network = copy.deepcopy(self.network)
        self.target_mixer = copy.deepcopy(self.mixer)

    def init_hidden(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.n_agents, self.hidden_dim, device=device)

    def initial_previous_actions(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.n_agents, self.action_dim, device=device)

    def step(self, obs: Tensor, previous_actions: Tensor, hidden: Tensor, *, target: bool = False,
             deterministic: bool = False, prune_threshold: float | None = None) -> MAICStep:
        """Advance the policy from ``obs: (B,n,O)`` and prior one-hot actions ``(B,n,A)``."""
        network = self.target_network if target else self.network
        return network(obs, previous_actions, hidden, deterministic=deterministic,
                       prune_threshold=prune_threshold)

    def teammate_model_loss(self, step: MAICStep, executed_actions: Tensor) -> Tensor:
        """Equation 1, averaged over the ``n * (n - 1)`` directed teammate pairs."""
        _batch, n, _, _ = step.mu.shape
        onehot = torch.nn.functional.one_hot(executed_actions, self.action_dim).to(step.mu.dtype)
        modeler_hidden = step.hidden.unsqueeze(2).expand(-1, -1, n, -1)
        modeled_action = onehot.unsqueeze(1).expand(-1, n, -1, -1)
        posterior_stats = self.network.inference_net(torch.cat((modeler_hidden, modeled_action), dim=-1))
        posterior_mu, posterior_sigma = self.network._gaussian(posterior_stats)
        model = torch.distributions.Normal(step.mu, step.sigma)
        posterior = torch.distributions.Normal(posterior_mu, posterior_sigma)
        kl = torch.distributions.kl_divergence(model, posterior).sum(-1)
        valid = ~torch.eye(n, device=kl.device, dtype=torch.bool).unsqueeze(0)
        return (kl * valid).sum(dim=(1, 2)) / (n * (n - 1))

    def sparsity_loss(self, step: MAICStep) -> Tensor:
        """Equation 4 using the same self-masked, scaled distribution as Equation 2.

        Base 2 and the 1e-4 clamp are the release's (``calculate_entropy_loss``); the
        paper leaves the base unspecified, and its tuned ``comm_beta`` is therefore in
        bits.  Using natural log here would weaken the term by ``1 / ln 2``.
        """
        alpha = self.network.attention(step.hidden.detach(), step.z.detach()).clamp_min(1e-4)
        return -(alpha * alpha.log2()).sum(dim=-1).mean(dim=1)

    def mix(self, chosen_q: Tensor, state: Tensor, *, target: bool = False) -> Tensor:
        """Aggregate ``chosen_q: (..., n)`` with configured VDN or QMIX."""
        if self.mixer_type == "vdn":
            return chosen_q.sum(dim=-1)
        mixer = self.target_mixer if target else self.mixer
        assert mixer is not None
        shape = chosen_q.shape[:-1]
        return mixer(chosen_q.reshape(-1, self.n_agents), state.reshape(-1, state.shape[-1])).view(*shape)

    def update_targets(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())


__all__ = ["MAICAgent", "MAICStep"]
