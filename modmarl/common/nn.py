"""Small neural-network constructors shared across algorithm modules.

Interface: ``build_mlp`` creates a ReLU MLP with explicit hidden widths and one linear
output layer. Algorithm-specific initialization and normalization stay with each model.
"""

from __future__ import annotations

from torch import nn


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


__all__ = ["build_mlp"]
