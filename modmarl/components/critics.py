"""Critics whose architecture is shared unchanged by several algorithms."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common.nn import build_mlp


class CentralizedMLPCritic(nn.Module):
    """Generic critic over flattened joint observations and actions."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = n_agents * (obs_dim + action_dim)
        self.net = build_mlp(input_dim, [hidden_dim, hidden_dim], 1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        flat = torch.cat([obs.reshape(obs.shape[0], -1), actions.reshape(actions.shape[0], -1)], dim=-1)
        return self.net(flat).squeeze(-1)


__all__ = ["CentralizedMLPCritic"]
