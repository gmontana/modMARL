"""Complete discrete-action HAPPO from Zhong et al. (JMLR 2024).

Model: every heterogeneous agent owns an independent local policy; one centralized
value network supplies a joint GAE estimate. Actors update in a fresh random order.
After each actor update, its new-to-old likelihood ratio multiplies the advantage
used by every actor that follows it.
Invariants: actors receive local observations only; the critic alone sees centralized
state; the sequential factor is frozen during one actor's PPO epochs and updated only
after that actor finishes; fresh on-policy data is discarded after one update cycle.
Interface: ``HAPPOAgent.act``, ``compute_gae``, and ``update`` contain the complete
learner. ``HAPPOBatch`` is the explicit rollout/update contract.
Why: the sequential likelihood factor is HAPPO's defining mechanism and remains next
to the policies, losses, optimizers, and value learner so the algorithm is readable in
one file.

References: Algorithm 4, Appendix C.4, and Tables 4/6 of the paper; official
``PKU-MARL/HARL`` revision b1af98b. Defaults reproduce its discrete MPE recipe:
128x128 orthogonal ReLU networks, feature normalization, Adam 5e-4/eps 1e-5,
five epochs, one minibatch, clip 0.2, ValueNorm, clipped Huber value loss, and
random agent order. Adaptation: modMARL environments expose concatenated local
observations instead of a separate global state; only the centralized critic input is
changed. The independent policies and sequential update are unchanged.
Table 7 is deliberately not used: it specifies the paper's off-policy HADDPG,
HATD3, MADDPG, and MATD3 experiments rather than HAPPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _orthogonal(linear: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)
    return linear


def _huber(error: Tensor, delta: float) -> Tensor:
    absolute = error.abs()
    return torch.where(absolute <= delta, 0.5 * error.square(), delta * (absolute - 0.5 * delta))


class ValueNorm(nn.Module):
    """Official debiased exponential value-target normalizer."""

    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.beta, self.epsilon = beta, epsilon
        self.register_buffer("running_mean", torch.zeros(()))
        self.register_buffer("running_mean_sq", torch.zeros(()))
        self.register_buffer("debiasing_term", torch.zeros(()))

    @torch.no_grad()
    def update(self, targets: Tensor) -> None:
        weight = self.beta
        self.running_mean.mul_(weight).add_(targets.mean() * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(targets.square().mean() * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 - weight)

    def statistics(self) -> tuple[Tensor, Tensor]:
        denominator = self.debiasing_term.clamp_min(self.epsilon)
        mean = self.running_mean / denominator
        variance = (self.running_mean_sq / denominator - mean.square()).clamp_min(1e-2)
        return mean, variance

    def normalize(self, values: Tensor) -> Tensor:
        mean, variance = self.statistics()
        return (values - mean) / variance.sqrt()

    def denormalize(self, values: Tensor) -> Tensor:
        mean, variance = self.statistics()
        return values * variance.sqrt() + mean


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, output_gain: float) -> None:
        super().__init__()
        gain = nn.init.calculate_gain("relu")
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = _orthogonal(nn.Linear(input_dim, hidden_dim), gain)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = _orthogonal(nn.Linear(hidden_dim, hidden_dim), gain)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output = _orthogonal(nn.Linear(hidden_dim, output_dim), output_gain)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.norm1(torch.relu(self.fc1(self.input_norm(inputs))))
        hidden = self.norm2(torch.relu(self.fc2(hidden)))
        return self.output(hidden)


class HAPPOActor(nn.Module):
    """One independent local categorical policy."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = _MLP(obs_dim, action_dim, hidden_dim, 0.01)

    def distribution(self, obs: Tensor, available_actions: Tensor | None = None) -> torch.distributions.Categorical:
        logits = self.network(obs)
        if available_actions is not None:
            logits = logits.masked_fill(~available_actions.bool(), torch.finfo(logits.dtype).min)
        return torch.distributions.Categorical(logits=logits)

    def forward(self, obs: Tensor) -> Tensor:
        return self.network(obs)

    def act(
        self, obs: Tensor, deterministic: bool = False, available_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(obs, available_actions)
        action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action)

    def evaluate_actions(
        self, obs: Tensor, actions: Tensor, available_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(obs, available_actions)
        return distribution.log_prob(actions), distribution.entropy()


class HAPPOValue(nn.Module):
    """Centralized scalar value baseline."""

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 128, state_dim: int | None = None) -> None:
        super().__init__()
        self.state_dim = state_dim if state_dim is not None else n_agents * obs_dim
        self.network = _MLP(self.state_dim, 1, hidden_dim, 1.0)

    def forward(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.state_dim:
            state = state.reshape(*state.shape[:-2], -1)
        return self.network(state).squeeze(-1)


@dataclass(frozen=True)
class HAPPOBatch:
    obs: Tensor
    states: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    active_masks: Tensor
    available_actions: Tensor


class HAPPOAgent(nn.Module):
    """Full HAPPO learner with independent actors and sequential PPO updates."""

    def __init__(
        self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int = 128,
        state_dim: int | None = None, *, actor_lr: float = 5e-4, critic_lr: float = 5e-4,
        gamma: float = 0.99, gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01, value_loss_coef: float = 1.0,
        max_grad_norm: float = 10.0, huber_delta: float = 10.0,
    ) -> None:
        super().__init__()
        self.n_agents, self.action_dim = n_agents, action_dim
        self.actors = nn.ModuleList(HAPPOActor(obs_dim, action_dim, hidden_dim) for _ in range(n_agents))
        self.critic = HAPPOValue(n_agents, obs_dim, hidden_dim, state_dim)
        self.value_normalizer = ValueNorm()
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5) for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_epsilon, self.entropy_coef = clip_epsilon, entropy_coef
        self.value_loss_coef, self.max_grad_norm = value_loss_coef, max_grad_norm
        self.huber_delta = huber_delta

    @torch.no_grad()
    def act(
        self, obs: Tensor, deterministic: bool = False,
        available_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        outcomes = [
            actor.act(obs[index], deterministic, None if available_actions is None else available_actions[index])
            for index, actor in enumerate(self.actors)
        ]
        actions, log_probs = zip(*outcomes)
        return torch.stack(actions), torch.stack(log_probs)

    def compute_gae(
        self, rewards: Tensor, normalized_values: Tensor, bootstrap_value: Tensor,
        transition_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values = self.value_normalizer.denormalize(normalized_values)
        next_value = self.value_normalizer.denormalize(bootstrap_value)
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(next_value)
        for step in range(rewards.shape[0] - 1, -1, -1):
            delta = rewards[step] + self.gamma * next_value * transition_masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * transition_masks[step] * gae
            advantages[step] = gae
            next_value = values[step]
        return advantages, advantages + values

    def update(
        self, batch: HAPPOBatch, *, actor_epochs: int = 5, critic_epochs: int = 5,
        actor_minibatches: int = 1, critic_minibatches: int = 1,
        order: Tensor | None = None,
    ) -> dict[str, object]:
        """Apply Algorithm 4: sequential actors first, then the centralized critic."""
        order = torch.randperm(self.n_agents, device=batch.obs.device) if order is None else order
        factor = torch.ones_like(batch.advantages)
        actor_losses, entropies = [], []
        for agent_index_tensor in order:
            agent_index = int(agent_index_tensor)
            actor, optimizer = self.actors[agent_index], self.actor_optimizers[agent_index]
            active = batch.active_masks[:, agent_index]
            advantages = batch.advantages.clone()
            selected = active.bool()
            if not selected.any():
                continue
            advantages = (advantages - advantages[selected].mean()) / (
                advantages[selected].std(unbiased=False) + 1e-5
            )
            old_log_probs = batch.old_log_probs[:, agent_index]
            for _ in range(actor_epochs):
                for indices in torch.randperm(batch.obs.shape[0], device=batch.obs.device).chunk(actor_minibatches):
                    log_probs, entropy = actor.evaluate_actions(
                        batch.obs[indices, agent_index], batch.actions[indices, agent_index],
                        batch.available_actions[indices, agent_index],
                    )
                    ratio = (log_probs - old_log_probs[indices]).exp()
                    weighted = factor[indices] * advantages[indices]
                    surrogate = torch.minimum(
                        ratio * weighted,
                        ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * weighted,
                    )
                    weights = active[indices]
                    policy_loss = -(surrogate * weights).sum() / weights.sum().clamp_min(1.0)
                    entropy_mean = (entropy * weights).sum() / weights.sum().clamp_min(1.0)
                    optimizer.zero_grad(set_to_none=True)
                    (policy_loss - self.entropy_coef * entropy_mean).backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    optimizer.step()
                    actor_losses.append(policy_loss.detach())
                    entropies.append(entropy_mean.detach())
            with torch.no_grad():
                new_log_probs = actor.evaluate_actions(
                    batch.obs[:, agent_index], batch.actions[:, agent_index],
                    batch.available_actions[:, agent_index],
                )[0]
                factor = factor * (new_log_probs - old_log_probs).exp()

        value_losses = []
        for _ in range(critic_epochs):
            for indices in torch.randperm(batch.states.shape[0], device=batch.states.device).chunk(critic_minibatches):
                self.value_normalizer.update(batch.returns[indices])
                values = self.critic(batch.states[indices])
                clipped = batch.old_values[indices] + (
                    values - batch.old_values[indices]
                ).clamp(-self.clip_epsilon, self.clip_epsilon)
                targets = self.value_normalizer.normalize(batch.returns[indices])
                loss = torch.maximum(
                    _huber(targets - values, self.huber_delta),
                    _huber(targets - clipped, self.huber_delta),
                ).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_loss_coef * loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                value_losses.append(loss.detach())
        return {
            "order": order.detach().clone(), "factor": factor.detach(),
            "policy_loss": float(torch.stack(actor_losses).mean()),
            "value_loss": float(torch.stack(value_losses).mean()),
            "entropy": float(torch.stack(entropies).mean()),
        }


__all__ = ["HAPPOActor", "HAPPOAgent", "HAPPOBatch", "HAPPOValue", "ValueNorm"]
