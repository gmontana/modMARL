"""Shared target-network parameter updates.

Invariants: updates run without gradient tracking and preserve parameter registration.
Interface: ``soft_update_module`` applies the standard Polyak interpolation in place.
"""

from __future__ import annotations

import torch


def soft_update_module(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)
