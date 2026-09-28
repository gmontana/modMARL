"""Who2Com's task-agnostic three-stage handshake.

Model: one requester broadcasts a compact query, supporters answer with local
keys, and general attention scores each query-key pair. Training receives the
softmax-weighted supporter feature; deterministic execution receives exactly the
highest-scoring supporter's feature and concatenates it with the requester's own.
Invariants: supporters never include the requester; tensors use
``(batch, supporters, feature)`` ordering; fusion concatenates along the feature
or channel dimension.
Interface: ``Who2ComHandshake.forward`` implements paper Equations 2--6 and
returns ``Who2ComOutput`` diagnostics.
Why: Who2Com is a supervised collaborative-perception component, explicitly not
a MARL learner. This dependency-free component follows the ICRA 2020 paper and
official ``GT-RIPL/MultiAgentPerception@4ef300547a7f7af2676a034f7cf742b009f57d99``;
the release's biased query projection is omitted because Equation 3 is bilinear.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Who2ComOutput:
    """Fused requester/supporter features and the selected communication link."""

    fused: Tensor
    received: Tensor
    attention: Tensor
    supporter_index: Tensor


class Who2ComHandshake(nn.Module):
    """General-attention request, match, and top-one connect mechanism.

    ``own_features`` has shape ``(B, F...)``, ``query`` is ``(B, Q)``,
    ``keys`` is ``(B, S, K)``, and ``supporter_features`` is ``(B, S, F...)``.
    The returned fused tensor has twice the size of dimension 1, matching the
    paper's channel-wise ``[requester; supporter]`` concatenation.
    """

    def __init__(self, query_dim: int, key_dim: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(query_dim, key_dim, bias=False)

    def forward(
        self,
        own_features: Tensor,
        query: Tensor,
        keys: Tensor,
        supporter_features: Tensor,
        *,
        hard: bool = False,
    ) -> Who2ComOutput:
        """Fuse one requester with soft training or hard execution support."""
        self._validate(own_features, query, keys, supporter_features)
        projected_query = self.query_projection(query)
        scores = torch.einsum("bsk,bk->bs", keys, projected_query)
        soft_attention = torch.softmax(scores, dim=1)
        supporter_index = scores.argmax(dim=1)
        attention = (
            torch.nn.functional.one_hot(
                supporter_index, num_classes=keys.shape[1],
            ).to(dtype=scores.dtype)
            if hard
            else soft_attention
        )
        feature_weights = attention.view(
            attention.shape[0], attention.shape[1],
            *([1] * (supporter_features.ndim - 2)),
        )
        received = (feature_weights * supporter_features).sum(dim=1)
        fused = torch.cat([own_features, received], dim=1)
        return Who2ComOutput(fused, received, attention, supporter_index)

    @staticmethod
    def _validate(
        own_features: Tensor,
        query: Tensor,
        keys: Tensor,
        supporter_features: Tensor,
    ) -> None:
        if query.ndim != 2 or keys.ndim != 3:
            raise ValueError("query must be (B,Q) and keys must be (B,S,K)")
        if keys.shape[1] < 1:
            raise ValueError("Who2Com requires at least one supporter")
        if own_features.ndim < 2 or supporter_features.ndim != own_features.ndim + 1:
            raise ValueError("supporter features must add one supporter axis")
        if not (
            own_features.shape[0]
            == query.shape[0]
            == keys.shape[0]
            == supporter_features.shape[0]
        ):
            raise ValueError("all Who2Com inputs must share a batch dimension")
        if keys.shape[1] != supporter_features.shape[1]:
            raise ValueError("keys and supporter features must list the same supporters")
        if own_features.shape[1:] != supporter_features.shape[2:]:
            raise ValueError("requester and supporter feature shapes must match")


__all__ = ["Who2ComHandshake", "Who2ComOutput"]
