"""Shared categorical-policy building blocks.

Model: a small discrete actor delegates categorical sampling to one functional helper.
Invariants: deterministic actions are exact argmax one-hots; stochastic actions preserve
the straight-through Gumbel path requested by the caller. Interface: ``DiscreteMLPActor``
and ``gumbel_policy_sample``. Why: several off-policy algorithms need the same sampling
contract without sharing an algorithm-specific policy architecture.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..common.nn import build_mlp


class DiscreteMLPActor(nn.Module):
    """Two-layer discrete actor with Gumbel-based sampling for categorical actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = build_mlp(obs_dim, [hidden_dim, hidden_dim], action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)

    def sample(
        self,
        obs: Tensor,
        temperature: float = 1.0,
        hard: bool = True,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self(obs)
        return gumbel_policy_sample(
            logits,
            action_dim=self.action_dim,
            temperature=temperature,
            hard=hard,
            deterministic=deterministic,
        )


def gumbel_policy_sample(
    logits: Tensor,
    *,
    action_dim: int,
    temperature: float = 1.0,
    hard: bool = True,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    if deterministic:
        action_idx = logits.argmax(dim=-1)
        action_one_hot = F.one_hot(action_idx, num_classes=action_dim).to(dtype=logits.dtype)
    else:
        action_one_hot = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        action_idx = action_one_hot.argmax(dim=-1)
    return action_one_hot, action_idx, logits
