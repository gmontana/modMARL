"""Generic on-policy return calculation and flat rollout storage.

Model: transitions are accumulated by episode and flattened after GAE is computed.
Invariants: true terminations use a zero bootstrap; truncations pass the caller's value.
Interface: ``compute_gae`` and ``FlatRollout`` contain no algorithm-specific policy logic.
Why: GAE and mechanical rollout storage are genuinely shared by several policy methods.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor | float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GAE(lambda) advantages and value targets along tensor axis zero."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(values[0])
    next_value = torch.as_tensor(bootstrap_value, dtype=values.dtype, device=values.device)
    for step in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[step] + gamma * next_value - values[step]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        advantages[step] = last_advantage
        next_value = values[step]
    return advantages, advantages + values


class FlatRollout:
    """Store feed-forward on-policy samples and compute GAE at episode boundaries."""

    def __init__(self, gamma: float, gae_lambda: float) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.advantages: list[torch.Tensor] = []
        self.returns: list[torch.Tensor] = []
        self._start_episode()

    def _start_episode(self) -> None:
        self._obs: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
    ) -> None:
        self._obs.append(obs)
        self._actions.append(actions)
        self._log_probs.append(log_probs)
        self._values.append(value)
        self._rewards.append(reward)

    def finish_episode(self, bootstrap_value: torch.Tensor | float) -> None:
        values = torch.stack(self._values)
        rewards = torch.stack(self._rewards)
        advantages, returns = compute_gae(
            rewards, values, bootstrap_value, self.gamma, self.gae_lambda,
        )
        self.obs.extend(self._obs)
        self.actions.extend(self._actions)
        self.log_probs.extend(self._log_probs)
        self.advantages.append(advantages)
        self.returns.append(returns)
        self._start_episode()

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.cat(self.advantages),
            torch.cat(self.returns),
        )


__all__ = ["FlatRollout", "compute_gae"]
