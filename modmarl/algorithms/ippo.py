"""Independent PPO as evaluated by de Witt et al. (2020).

Model: one parameter-shared local actor and one parameter-shared local value
function are applied independently to every agent observation.  A rollout stores
the behaviour log-probability and value for every agent, then computes a separate
GAE(lambda) target for each agent from the shared team reward.
Invariants: neither actor nor critic receives another agent's observation; PPO
ratios and value clipping are elementwise per agent; advantages are normalised
once over the complete rollout before minibatching.
Interface: ``IPPOAgent.act`` collects on-policy data and ``IPPOAgent.update`` owns
the complete paper loss and optimiser step.  ``IPPORollout`` owns trajectory
boundaries and truncation bootstrapping.
Why: keeping the learner here makes the defining independent PPO procedure
auditable without following an algorithm-specific trainer in another file.

Paper: C. S. de Witt et al., "Is Independent Learning All You Need in the
StarCraft Multi-Agent Challenge?", arXiv:2011.09533, equations 4--7.
Reference implementation: Denys Makoviichuk's ``rl_games`` PPO/SMAC release.
The default MLP is the paper's vector-observation architecture (256, 128).  Its
SMAC-only Conv1d variants are not selected for vector environments, and the runner
collects the actor batch sequentially because modMARL's small environments are not
vectorized.  This preserves local information and on-policy aggregation.  Observation
normalization remains off, matching the paper's 2m_vs_1z MLP configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..common.nn import build_mlp
from ..common.on_policy import compute_gae


def _paper_initialise(module: nn.Module) -> None:
    """Released variance-scaling initializer: truncated N(0, 2/fan_in)."""
    if isinstance(module, nn.Linear):
        standard_deviation = math.sqrt(2.0 / module.in_features)
        nn.init.trunc_normal_(
            module.weight, std=standard_deviation,
            a=-2.0 * standard_deviation, b=2.0 * standard_deviation,
        )
        nn.init.zeros_(module.bias)


@dataclass(frozen=True)
class IPPOBatch:
    """Flattened on-policy samples; leading dimensions are time and agent."""

    obs: Tensor
    available_actions: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor


class IPPORollout:
    """Fresh trajectories used once by IPPO, preserving episode boundaries."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._episodes: list[IPPOBatch] = []
        self._obs: list[Tensor] = []
        self._available_actions: list[Tensor] = []
        self._actions: list[Tensor] = []
        self._log_probs: list[Tensor] = []
        self._values: list[Tensor] = []
        self._rewards: list[Tensor] = []

    def add(
        self, obs: Tensor, available_actions: Tensor, actions: Tensor,
        log_probs: Tensor, values: Tensor, team_reward: float,
    ) -> None:
        """Append one joint step; per-agent tensors have leading shape ``(n_agents,)``."""
        self._obs.append(obs.detach())
        self._available_actions.append(available_actions.detach())
        self._actions.append(actions.detach())
        self._log_probs.append(log_probs.detach())
        self._values.append(values.detach())
        self._rewards.append(torch.full_like(values, float(team_reward)))

    def finish_episode(self, bootstrap_value: Tensor) -> None:
        """Close an episode, using zero bootstrap for termination and V(o_T) for truncation."""
        if not self._obs:
            raise ValueError("cannot finish an empty IPPO episode")
        values = torch.stack(self._values)
        rewards = torch.stack(self._rewards)
        advantages, returns = compute_gae(
            rewards, values, bootstrap_value, self.gamma, self.gae_lambda,
        )
        self._episodes.append(
            IPPOBatch(
                obs=torch.stack(self._obs),
                available_actions=torch.stack(self._available_actions),
                actions=torch.stack(self._actions),
                old_log_probs=torch.stack(self._log_probs),
                old_values=values,
                advantages=advantages,
                returns=returns,
            )
        )
        self._obs.clear()
        self._available_actions.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()

    def batch(self) -> IPPOBatch:
        """Concatenate episodes, then flatten time-agent pairs as in the release."""
        if self._obs:
            raise RuntimeError("finish the current episode before requesting the rollout")
        if not self._episodes:
            raise RuntimeError("the IPPO rollout contains no completed episodes")
        concatenated = {
            field: torch.cat([getattr(episode, field) for episode in self._episodes], dim=0)
            for field in IPPOBatch.__dataclass_fields__
        }
        return IPPOBatch(
            obs=concatenated["obs"].flatten(0, 1),
            available_actions=concatenated["available_actions"].flatten(0, 1),
            actions=concatenated["actions"].flatten(),
            old_log_probs=concatenated["old_log_probs"].flatten(),
            old_values=concatenated["old_values"].flatten(),
            advantages=concatenated["advantages"].flatten(),
            returns=concatenated["returns"].flatten(),
        )

class IPPOActor(nn.Module):
    """Parameter-shared local categorical policy pi(a_i | o_i)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, list(hidden_dims), action_dim)
        self.net.apply(_paper_initialise)

    def forward(self, obs: Tensor, available_actions: Tensor | None = None) -> Tensor:
        logits = self.net(obs)
        if available_actions is not None:
            logits = logits.masked_fill(~available_actions.bool(), torch.finfo(logits.dtype).min)
        return logits

    def distribution(self, obs: Tensor, available_actions: Tensor | None = None) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(obs, available_actions))


class LocalValue(nn.Module):
    """Parameter-shared local critic V(o_i), with no centralized inputs."""

    def __init__(self, obs_dim: int, hidden_dims: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, list(hidden_dims), 1)
        self.net.apply(_paper_initialise)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)


class IPPOAgent(nn.Module):
    """Complete independent PPO learner from equations 4--7 of the paper."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int] = (256, 128),
        learning_rate: float = 1e-4,
        clip_epsilon: float = 0.2,
        critic_coef: float = 1.0,
        entropy_coef: float = 0.005,
        max_grad_norm: float = 0.5,
    ) -> None:
        super().__init__()
        self.actor = IPPOActor(obs_dim, action_dim, hidden_dims)
        self.critic = LocalValue(obs_dim, hidden_dims)
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(
        self, obs: Tensor, available_actions: Tensor | None = None, deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-agent actions, behaviour log-probabilities, and local values."""
        distribution = self.actor.distribution(obs, available_actions)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), self.critic(obs)

    def losses(self, batch: IPPOBatch, index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Equations 5--7 for one minibatch, including clipped value regression."""
        obs = batch.obs[index]
        actions = batch.actions[index]
        old_log_probs = batch.old_log_probs[index]
        old_values = batch.old_values[index]
        advantages = batch.advantages[index]
        returns = batch.returns[index]

        distribution = self.actor.distribution(obs, batch.available_actions[index])
        log_probs = distribution.log_prob(actions)
        ratio = (log_probs - old_log_probs).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        values = self.critic(obs)
        clipped_values = old_values + (values - old_values).clamp(-self.clip_epsilon, self.clip_epsilon)
        # Equation 6 prints ``min``, but rl_games and standard PPO use ``max``: clipping
        # must not let the critic obtain an artificially smaller loss by moving too far.
        value_loss = torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
        entropy = distribution.entropy().mean()
        return policy_loss, value_loss, entropy

    def update(self, rollout: IPPORollout, epochs: int = 4, minibatch_size: int = 1024) -> dict[str, float]:
        """Consume one fresh rollout using paper-wide advantage normalization."""
        batch = rollout.batch()
        advantages = batch.advantages
        normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        batch = IPPOBatch(
            obs=batch.obs,
            available_actions=batch.available_actions,
            actions=batch.actions,
            old_log_probs=batch.old_log_probs,
            old_values=batch.old_values,
            advantages=normalized,
            returns=batch.returns,
        )
        sample_count = batch.obs.shape[0]
        totals = torch.zeros(3, device=batch.obs.device)
        updates = 0
        for _ in range(epochs):
            permutation = torch.randperm(sample_count, device=batch.obs.device)
            for start in range(0, sample_count, minibatch_size):
                index = permutation[start : start + minibatch_size]
                policy_loss, value_loss, entropy = self.losses(batch, index)
                loss = policy_loss + self.critic_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                totals += torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach()))
                updates += 1
        means = totals / updates
        return {"policy_loss": float(means[0]), "value_loss": float(means[1]), "entropy": float(means[2])}


__all__ = ["IPPOActor", "IPPOAgent", "IPPOBatch", "IPPORollout", "LocalValue"]
