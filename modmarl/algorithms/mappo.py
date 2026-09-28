"""Recurrent MAPPO from Yu et al. (NeurIPS 2022 Datasets and Benchmarks).

Model: homogeneous agents share a recurrent local categorical actor and a
recurrent centralized value function.  For the paper's MPE setting, each critic
input is the concatenation of all local observations.  Fresh trajectories are
split into ten-step chunks for BPTT, and all agents' samples train shared weights.
Invariants: the actor never receives centralized information; stored recurrent
states begin each training chunk; termination resets recurrence while time-limit
truncation bootstraps the centralized value.
Interface: ``MAPPOAgent.act`` advances actor and critic states, and
``MAPPOAgent.update`` owns GAE targets, PPO/value clipping, value normalization,
Huber loss, optimizers, gradient clipping, and recurrent minibatching.
Why: MAPPO's published strength depends on these implementation details, so the
algorithm-specific path is kept together rather than hidden in an example script.

References: main paper Sections 3 and 5; supplement Algorithm 1, Table 4 (common
MAPPO settings), Table 6 (MPE settings), and Table 13 (adopted MPE settings);
official ``marlbenchmark/on-policy`` revision de66d7a.  Defaults select
the published MPE Spread variant: shared 64-wide Tanh recurrent networks, 7e-4
Adam, ten epochs, one minibatch, value normalization, and ten-step BPTT chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _orthogonal(linear: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)
    return linear


def huber_loss(error: Tensor, delta: float = 10.0) -> Tensor:
    """Official piecewise Huber loss, without an implicit reduction."""
    absolute = error.abs()
    return torch.where(absolute <= delta, 0.5 * error.square(), delta * (absolute - 0.5 * delta))


class ValueNorm(nn.Module):
    """Debiased exponential value-target normalizer from the official release."""

    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.beta, self.epsilon = beta, epsilon
        self.register_buffer("running_mean", torch.zeros(()))
        self.register_buffer("running_mean_sq", torch.zeros(()))
        self.register_buffer("debiasing_term", torch.zeros(()))

    @torch.no_grad()
    def update(self, targets: Tensor) -> None:
        mean, mean_sq = targets.mean(), targets.square().mean()
        self.running_mean.mul_(self.beta).add_(mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(mean_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def mean_variance(self) -> tuple[Tensor, Tensor]:
        denominator = self.debiasing_term.clamp_min(self.epsilon)
        mean = self.running_mean / denominator
        variance = (self.running_mean_sq / denominator - mean.square()).clamp_min(1e-2)
        return mean, variance

    def normalize(self, targets: Tensor) -> Tensor:
        mean, variance = self.mean_variance()
        return (targets - mean) / variance.sqrt()

    def denormalize(self, values: Tensor) -> Tensor:
        mean, variance = self.mean_variance()
        return values * variance.sqrt() + mean


class RecurrentBackbone(nn.Module):
    """Feature LayerNorm, two orthogonal Tanh layers, GRU, and output LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        gain = nn.init.calculate_gain("tanh")
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = _orthogonal(nn.Linear(input_dim, hidden_dim), gain)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = _orthogonal(nn.Linear(hidden_dim, hidden_dim), gain)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        for name, parameter in self.gru.named_parameters():
            nn.init.zeros_(parameter) if "bias" in name else nn.init.orthogonal_(parameter)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: Tensor, hidden: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        features = self.norm1(torch.tanh(self.fc1(self.input_norm(inputs))))
        features = self.norm2(torch.tanh(self.fc2(features)))
        hidden = self.gru(features, hidden * masks.unsqueeze(-1))
        return self.output_norm(hidden), hidden


class MAPPOActor(nn.Module):
    """Shared recurrent decentralized policy pi(a_i | o_i, h_i)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, gain: float = 0.01) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = RecurrentBackbone(obs_dim, hidden_dim)
        self.action_head = _orthogonal(nn.Linear(hidden_dim, action_dim), gain)

    def step(
        self, obs: Tensor, hidden: Tensor, masks: Tensor,
        available_actions: Tensor | None = None,
    ) -> tuple[torch.distributions.Categorical, Tensor]:
        features, hidden = self.backbone(obs, hidden, masks)
        logits = self.action_head(features)
        if available_actions is not None:
            logits = logits.masked_fill(~available_actions.bool(), torch.finfo(logits.dtype).min)
        return torch.distributions.Categorical(logits=logits), hidden


class CentralizedValue(nn.Module):
    """Shared recurrent critic V(s_i, h_i^V); state may be agent-specific."""

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = RecurrentBackbone(state_dim, hidden_dim)
        self.value_head = _orthogonal(nn.Linear(hidden_dim, 1), 1.0)

    def step(self, state: Tensor, hidden: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        features, hidden = self.backbone(state, hidden, masks)
        return self.value_head(features).squeeze(-1), hidden


@dataclass(frozen=True)
class MAPPOBatch:
    obs: Tensor
    states: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    actor_hidden: Tensor
    critic_hidden: Tensor
    masks: Tensor
    active_masks: Tensor
    available_actions: Tensor
    valid: Tensor


class MAPPORollout:
    """Fresh recurrent trajectories, converted to padded paper-length BPTT chunks."""

    _FIELDS = (
        "obs", "states", "actions", "old_log_probs", "old_values", "actor_hidden",
        "critic_hidden", "masks", "active_masks", "available_actions",
    )

    def __init__(self, learner: MAPPOAgent) -> None:
        self.learner = learner
        self.episodes: list[MAPPOBatch] = []
        self._current: dict[str, list[Tensor]] = {field: [] for field in self._FIELDS}
        self._rewards: list[Tensor] = []

    def add(
        self, *, obs: Tensor, states: Tensor, actions: Tensor, log_probs: Tensor,
        values: Tensor, actor_hidden: Tensor, critic_hidden: Tensor, masks: Tensor,
        team_reward: float, active_masks: Tensor | None = None,
        available_actions: Tensor | None = None,
    ) -> None:
        if active_masks is None:
            active_masks = torch.ones_like(masks)
        if available_actions is None:
            available_actions = torch.ones(
                (*actions.shape, self.learner.action_dim), device=actions.device, dtype=torch.bool,
            )
        values_by_field = (
            obs, states, actions, log_probs, values, actor_hidden, critic_hidden, masks,
            active_masks, available_actions,
        )
        for field, value in zip(self._FIELDS, values_by_field):
            self._current[field].append(value.detach())
        self._rewards.append(torch.full_like(values, float(team_reward)))

    def finish_episode(self, bootstrap_value: Tensor, final_mask: Tensor) -> None:
        if not self._rewards:
            raise ValueError("cannot finish an empty MAPPO episode")
        values = torch.stack(self._current["old_values"])
        rewards = torch.stack(self._rewards)
        transition_masks = torch.stack(self._current["masks"][1:] + [final_mask])
        advantages, returns = self.learner.compute_gae(rewards, values, bootstrap_value, transition_masks)
        tensors = {field: torch.stack(items) for field, items in self._current.items()}
        tensors.update(advantages=advantages, returns=returns, valid=torch.ones_like(advantages))
        self.episodes.append(MAPPOBatch(**tensors))
        self._current = {field: [] for field in self._FIELDS}
        self._rewards.clear()

    def batch(self) -> MAPPOBatch:
        """Split every agent trajectory into independently shuffled recurrent chunks."""
        if self._rewards or not self.episodes:
            raise RuntimeError("finish all MAPPO episodes before requesting a batch")
        chunks: dict[str, list[Tensor]] = {field: [] for field in MAPPOBatch.__dataclass_fields__}
        length = self.learner.chunk_length
        for episode in self.episodes:
            horizon, agents = episode.obs.shape[:2]
            for agent in range(agents):
                for start in range(0, horizon, length):
                    stop = min(start + length, horizon)
                    count = stop - start
                    for field in ("obs", "states", "available_actions"):
                        value = getattr(episode, field)[start:stop, agent]
                        padding = torch.zeros(
                            (length - count, *value.shape[1:]), device=value.device, dtype=value.dtype,
                        )
                        chunks[field].append(torch.cat((value, padding)))
                    for field in (
                        "actions", "old_log_probs", "old_values", "advantages", "returns",
                        "masks", "active_masks", "valid",
                    ):
                        value = getattr(episode, field)[start:stop, agent]
                        padding = torch.zeros(length - count, device=value.device, dtype=value.dtype)
                        chunks[field].append(torch.cat((value, padding)))
                    chunks["actor_hidden"].append(episode.actor_hidden[start, agent])
                    chunks["critic_hidden"].append(episode.critic_hidden[start, agent])
        return MAPPOBatch(**{field: torch.stack(values) for field, values in chunks.items()})


class MAPPOAgent(nn.Module):
    """Complete recurrent MAPPO learner for homogeneous discrete-action teams."""

    def __init__(
        self, n_agents: int, obs_dim: int, action_dim: int, state_dim: int | None = None,
        hidden_dim: int = 64, *, actor_lr: float = 7e-4, critic_lr: float = 7e-4,
        clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0, max_grad_norm: float = 10.0,
        huber_delta: float = 10.0, gamma: float = 0.99, gae_lambda: float = 0.95,
        chunk_length: int = 10,
    ) -> None:
        super().__init__()
        self.n_agents, self.obs_dim, self.action_dim = n_agents, obs_dim, action_dim
        self.state_dim = state_dim if state_dim is not None else n_agents * obs_dim
        self.actor = MAPPOActor(obs_dim, action_dim, hidden_dim)
        self.critic = CentralizedValue(self.state_dim, hidden_dim)
        self.value_normalizer = ValueNorm()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.clip_epsilon, self.entropy_coef = clip_epsilon, entropy_coef
        self.value_loss_coef, self.max_grad_norm = value_loss_coef, max_grad_norm
        self.huber_delta, self.gamma, self.gae_lambda = huber_delta, gamma, gae_lambda
        self.chunk_length = chunk_length

    def initial_state(self, device: torch.device) -> tuple[Tensor, Tensor]:
        shape = (self.n_agents, self.actor.hidden_dim)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @torch.no_grad()
    def act(
        self, obs: Tensor, states: Tensor, actor_hidden: Tensor, critic_hidden: Tensor,
        masks: Tensor, available_actions: Tensor | None = None, deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        distribution, actor_hidden = self.actor.step(obs, actor_hidden, masks, available_actions)
        actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        values, critic_hidden = self.critic.step(states, critic_hidden, masks)
        return actions, distribution.log_prob(actions), values, actor_hidden, critic_hidden

    def compute_gae(
        self, rewards: Tensor, normalized_values: Tensor, bootstrap_value: Tensor, masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """GAE using denormalized critic outputs, independently per agent."""
        values = self.value_normalizer.denormalize(normalized_values)
        next_value = self.value_normalizer.denormalize(bootstrap_value)
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(next_value)
        for step in range(rewards.shape[0] - 1, -1, -1):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
            next_value = values[step]
        return advantages, advantages + values

    def _unroll(self, batch: MAPPOBatch, index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        actor_hidden = batch.actor_hidden[index]
        critic_hidden = batch.critic_hidden[index]
        log_probs, entropies, values = [], [], []
        for step in range(batch.obs.shape[1]):
            distribution, actor_hidden = self.actor.step(
                batch.obs[index, step], actor_hidden, batch.masks[index, step],
                batch.available_actions[index, step],
            )
            value, critic_hidden = self.critic.step(
                batch.states[index, step], critic_hidden, batch.masks[index, step],
            )
            log_probs.append(distribution.log_prob(batch.actions[index, step]))
            entropies.append(distribution.entropy())
            values.append(value)
        return torch.stack(log_probs, 1), torch.stack(entropies, 1), torch.stack(values, 1)

    def losses(self, batch: MAPPOBatch, index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        log_probs, entropies, values = self._unroll(batch, index)
        active = batch.valid[index] * batch.active_masks[index]
        ratio = (log_probs - batch.old_log_probs[index]).exp()
        advantages = batch.advantages[index]
        surrogate = torch.minimum(
            ratio * advantages,
            ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages,
        )
        policy_loss = -(surrogate * active).sum() / active.sum().clamp_min(1.0)
        entropy = (entropies * active).sum() / active.sum().clamp_min(1.0)

        clipped_values = batch.old_values[index] + (
            values - batch.old_values[index]
        ).clamp(-self.clip_epsilon, self.clip_epsilon)
        normalized_returns = self.value_normalizer.normalize(batch.returns[index])
        original = huber_loss(normalized_returns - values, self.huber_delta)
        clipped = huber_loss(normalized_returns - clipped_values, self.huber_delta)
        value_loss = (torch.maximum(original, clipped) * active).sum() / active.sum().clamp_min(1.0)
        return policy_loss, value_loss, entropy

    def update(self, batch: MAPPOBatch, epochs: int = 10, num_minibatches: int = 1) -> dict[str, float]:
        active = batch.valid.bool() & batch.active_masks.bool()
        valid_advantages = batch.advantages[active]
        normalized = (batch.advantages - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-5)
        batch = MAPPOBatch(**{**batch.__dict__, "advantages": normalized})
        chunk_count = batch.obs.shape[0]
        totals = torch.zeros(3, device=batch.obs.device)
        updates = 0
        for _ in range(epochs):
            permutation = torch.randperm(chunk_count, device=batch.obs.device)
            for index in permutation.chunk(num_minibatches):
                self.value_normalizer.update(batch.returns[index][batch.valid[index].bool()])
                policy_loss, value_loss, entropy = self.losses(batch, index)
                self.actor_optimizer.zero_grad(set_to_none=True)
                (policy_loss - self.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                totals += torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach()))
                updates += 1
        means = totals / updates
        return {"policy_loss": float(means[0]), "value_loss": float(means[1]), "entropy": float(means[2])}


__all__ = [
    "CentralizedValue", "MAPPOActor", "MAPPOAgent", "MAPPOBatch", "MAPPORollout", "RecurrentBackbone",
    "ValueNorm", "huber_loss",
]
