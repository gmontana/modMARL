from __future__ import annotations

import pytest
import torch

from modmarl import Who2ComHandshake, Who2ComOutput


def test_who2com_general_attention_matches_equation_three() -> None:
    handshake = Who2ComHandshake(query_dim=2, key_dim=2)
    with torch.no_grad():
        handshake.query_projection.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
    query = torch.tensor([[1.0, 2.0]])
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    own = torch.tensor([[10.0, 20.0]])
    supporters = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    output = handshake(own, query, keys, supporters)
    expected_scores = torch.tensor([[2.0, 2.0, 4.0]])
    expected_attention = torch.softmax(expected_scores, dim=1)
    expected_received = (expected_attention.unsqueeze(-1) * supporters).sum(dim=1)
    assert torch.allclose(output.attention, expected_attention)
    assert torch.allclose(output.received, expected_received)


def test_who2com_equation_four_concatenates_requester_then_supporter() -> None:
    handshake = Who2ComHandshake(query_dim=1, key_dim=1)
    with torch.no_grad():
        handshake.query_projection.weight.fill_(1.0)
    own = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    supporters = torch.tensor(
        [[[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]],
    )
    output = handshake(
        own, torch.ones(1, 1), torch.tensor([[[1.0], [2.0]]]),
        supporters, hard=True,
    )
    assert output.supporter_index.item() == 1
    assert torch.equal(output.fused[:, :2], own)
    assert torch.equal(output.fused[:, 2:], supporters[:, 1])


def test_who2com_execution_connects_exactly_one_supporter() -> None:
    torch.manual_seed(3)
    handshake = Who2ComHandshake(query_dim=3, key_dim=4)
    output = handshake(
        torch.randn(5, 6), torch.randn(5, 3), torch.randn(5, 4, 4),
        torch.randn(5, 4, 6), hard=True,
    )
    assert torch.all(output.attention.sum(dim=1) == 1)
    assert set(output.attention.flatten().tolist()) <= {0.0, 1.0}
    assert torch.equal(output.attention.argmax(dim=1), output.supporter_index)


def test_who2com_soft_handshake_has_complete_gradient_path() -> None:
    handshake = Who2ComHandshake(query_dim=3, key_dim=4)
    query = torch.randn(2, 3, requires_grad=True)
    keys = torch.randn(2, 3, 4, requires_grad=True)
    supporters = torch.randn(2, 3, 5, requires_grad=True)
    output = handshake(torch.randn(2, 5), query, keys, supporters)
    output.fused.square().mean().backward()
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert keys.grad is not None and keys.grad.abs().sum() > 0
    assert supporters.grad is not None and supporters.grad.abs().sum() > 0


def test_who2com_batch_one_and_single_supporter() -> None:
    handshake = Who2ComHandshake(query_dim=2, key_dim=3)
    output = handshake(
        torch.randn(1, 4), torch.randn(1, 2), torch.randn(1, 1, 3),
        torch.randn(1, 1, 4),
    )
    assert isinstance(output, Who2ComOutput)
    assert output.attention.item() == pytest.approx(1.0)
    assert output.fused.shape == (1, 8)


def test_who2com_rejects_an_empty_supporter_set() -> None:
    handshake = Who2ComHandshake(query_dim=2, key_dim=3)
    with pytest.raises(ValueError, match="at least one supporter"):
        handshake(
            torch.randn(1, 4), torch.randn(1, 2), torch.randn(1, 0, 3),
            torch.randn(1, 0, 4),
        )
