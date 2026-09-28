from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from modmarl.algorithms.commnet import CommNetCell
from modmarl.algorithms.ddpg import DDPGCritic as IndependentMLPCritic
from modmarl.algorithms.maac import AttentionCritic
from modmarl.algorithms.marc import DenseRGCNLayer
from modmarl.algorithms.marc import MARCRelationalCritic as RelationalCritic
from modmarl.components import (
    CentralizedMLPCritic,
    DiscreteMLPActor,
    soft_update_module,
)


def _assert_module_has_finite_gradients(module: nn.Module) -> None:
    grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert grads
    assert all(grad is None or torch.isfinite(grad).all() for grad in grads)
    assert any(grad is not None and grad.abs().sum().item() > 0.0 for grad in grads)


def test_commnet_cell_shapes() -> None:
    block = CommNetCell(obs_dim=5, hidden_dim=16)
    outputs = block(
        torch.randn(4, 3, 5),
        torch.randn(4, 3, 16),
        torch.randn(4, 3, 16),
        torch.randn(4, 3, 16),
        torch.ones(4, 3),
    )
    assert [output.shape for output in outputs] == [(4, 3, 16)] * 3


def test_centralized_critic_shapes() -> None:
    critic = CentralizedMLPCritic(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(6, 3, 5)
    actions = torch.randn(6, 3, 4)
    q_values = critic(obs, actions)
    assert tuple(q_values.shape) == (6,)


def test_independent_critic_shapes() -> None:
    critic = IndependentMLPCritic(obs_dim=5, action_dim=4, hidden_dims=(16, 12))
    obs = torch.randn(6, 5)
    actions = torch.randn(6, 4)
    q_values = critic(obs, actions)
    assert tuple(q_values.shape) == (6,)


def test_discrete_actor_shapes() -> None:
    actor = DiscreteMLPActor(obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(6, 5)
    one_hot, action_idx, logits = actor.sample(obs, deterministic=False)
    assert tuple(one_hot.shape) == (6, 4)
    assert tuple(action_idx.shape) == (6,)
    assert tuple(logits.shape) == (6, 4)


def test_attention_critic_shapes() -> None:
    critic = AttentionCritic(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16, attend_heads=4)
    obs = torch.randn(6, 3, 5)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (6, 3)), num_classes=4).to(dtype=torch.float32)
    output = critic(obs, actions, return_attention=True)
    assert tuple(output.q_taken.shape) == (6, 3)
    assert tuple(output.all_q.shape) == (6, 3, 4)
    assert output.attention is not None
    assert len(output.attention) == 3


def test_relational_critic_shapes() -> None:
    critic = RelationalCritic(
        n_agents=2,
        node_feature_dim=6,
        action_dim=4,
        hidden_dim=16,
        embed_dim=16,
        num_relations=6,
        num_relational_layers=1,
    )
    node_features = torch.randn(5, 2, 4, 6)
    relations = torch.randint(0, 2, (5, 6, 4, 4), dtype=torch.float32)
    actions = torch.nn.functional.one_hot(torch.randint(0, 4, (5, 2)), num_classes=4).to(dtype=torch.float32)
    q_taken, all_q = critic(node_features, relations, actions, return_all_q=True)
    assert tuple(q_taken.shape) == (5, 2)
    assert tuple(all_q.shape) == (5, 2, 4)


def test_soft_update_module_interpolates_parameters_exactly() -> None:
    source = nn.Linear(3, 2)
    target = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(8.0)
        source.bias.fill_(10.0)
        target.weight.fill_(2.0)
        target.bias.fill_(4.0)

    soft_update_module(target, source, tau=0.25)

    assert torch.allclose(target.weight, torch.full_like(target.weight, 3.5))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 5.5))


def test_discrete_actor_deterministic_sample_matches_argmax() -> None:
    actor = DiscreteMLPActor(obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(6, 5)
    logits = actor(obs)
    one_hot, action_idx, sampled_logits = actor.sample(obs, deterministic=True)

    expected_idx = logits.argmax(dim=-1)
    expected_one_hot = F.one_hot(expected_idx, num_classes=4).to(dtype=logits.dtype)
    assert torch.equal(action_idx, expected_idx)
    assert torch.equal(one_hot, expected_one_hot)
    assert torch.allclose(sampled_logits, logits)


def test_discrete_actor_backward_produces_finite_gradients() -> None:
    torch.manual_seed(0)
    actor = DiscreteMLPActor(obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(8, 5)
    action_one_hot, _, logits = actor.sample(obs, temperature=0.7, hard=False, deterministic=False)
    loss = (action_one_hot * logits).sum()
    loss.backward()
    _assert_module_has_finite_gradients(actor)


def test_centralized_critic_backward_produces_finite_gradients() -> None:
    torch.manual_seed(0)
    critic = CentralizedMLPCritic(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(6, 3, 5)
    actions = torch.randn(6, 3, 4)
    loss = critic(obs, actions).pow(2).mean()
    loss.backward()
    _assert_module_has_finite_gradients(critic)


def test_independent_critic_can_overfit_single_batch() -> None:
    torch.manual_seed(0)
    critic = IndependentMLPCritic(obs_dim=5, action_dim=4, hidden_dims=(32, 24))
    optimizer = torch.optim.Adam(critic.parameters(), lr=5e-2)
    obs = torch.randn(32, 5)
    actions = torch.randn(32, 4)
    targets = torch.randn(32)

    with torch.no_grad():
        initial_loss = F.mse_loss(critic(obs, actions), targets).item()

    for _ in range(200):
        loss = F.mse_loss(critic(obs, actions), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_loss = F.mse_loss(critic(obs, actions), targets).item()
    assert final_loss < initial_loss * 0.2


def test_centralized_critic_can_overfit_single_batch() -> None:
    torch.manual_seed(0)
    critic = CentralizedMLPCritic(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=32)
    optimizer = torch.optim.Adam(critic.parameters(), lr=5e-2)
    obs = torch.randn(32, 3, 5)
    actions = torch.randn(32, 3, 4)
    targets = torch.randn(32)

    with torch.no_grad():
        initial_loss = F.mse_loss(critic(obs, actions), targets).item()

    for _ in range(200):
        loss = F.mse_loss(critic(obs, actions), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_loss = F.mse_loss(critic(obs, actions), targets).item()
    assert final_loss < initial_loss * 0.2


def test_dense_rgcn_layer_aggregates_row_neighbours_per_relation() -> None:
    torch.manual_seed(0)
    layer = DenseRGCNLayer(input_dim=4, output_dim=4, num_relations=2)
    with torch.no_grad():
        layer.self_linear.weight.zero_()
        layer.relation_linears[0].weight.copy_(torch.eye(4))
        layer.relation_linears[1].weight.zero_()

    node_features = torch.randn(1, 3, 4)
    relations = torch.zeros(1, 2, 3, 3)
    relations[0, 0, 0, 2] = 1.0   # node 2 is a relation-0 neighbour OF node 0

    out = layer(node_features, relations)

    assert torch.allclose(out[0, 0], node_features[0, 2] + layer.bias)
    assert torch.allclose(out[0, 1], layer.bias)
    assert torch.allclose(out[0, 2], layer.bias)
