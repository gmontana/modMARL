from __future__ import annotations

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_marc import train
from marl_envs import MACPPEnv, macpp_available
from modmarl import MARCActor, MARCAgent, MARCRelationalCritic
from modmarl.algorithms.marc import DenseRGCNLayer


def test_marc_actor_shapes() -> None:
    actor = MARCActor(obs_dim=12, action_dim=6, hidden_dim=32)
    obs = torch.randn(4, 12)
    one_hot, action_idx, logits, probs, log_probs, chosen_log_prob, entropy = actor.sample(obs, deterministic=False)
    assert tuple(one_hot.shape) == (4, 6)
    assert tuple(action_idx.shape) == (4,)
    assert tuple(logits.shape) == (4, 6)
    assert tuple(probs.shape) == (4, 6)
    assert tuple(log_probs.shape) == (4, 6)
    assert tuple(chosen_log_prob.shape) == (4,)
    assert tuple(entropy.shape) == (4,)


def test_marc_actor_deterministic_sample_matches_argmax() -> None:
    actor = MARCActor(obs_dim=12, action_dim=6, hidden_dim=32)
    obs = torch.randn(4, 12)
    logits = actor(obs)
    one_hot, action_idx, sampled_logits, _, log_probs, chosen_log_prob, _ = actor.sample(obs, deterministic=True)
    expected_idx = logits.argmax(dim=-1)
    expected_one_hot = torch.nn.functional.one_hot(expected_idx, num_classes=6).to(dtype=logits.dtype)
    expected_log_prob = log_probs.gather(-1, expected_idx.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(action_idx, expected_idx)
    assert torch.equal(one_hot, expected_one_hot)
    assert torch.allclose(sampled_logits, logits)
    assert torch.allclose(chosen_log_prob, expected_log_prob)


def test_marc_critic_shapes() -> None:
    critic = MARCRelationalCritic(
        n_agents=2,
        node_feature_dim=6,
        action_dim=6,
        hidden_dim=32,
        embed_dim=32,
        num_relations=6,
        num_relational_layers=2,
    )
    node_features = torch.randn(5, 2, 4, 6)
    relations = torch.randint(0, 2, (5, 6, 4, 4), dtype=torch.float32)
    actions = torch.nn.functional.one_hot(torch.randint(0, 6, (5, 2)), num_classes=6).to(dtype=torch.float32)
    q_taken, all_q = critic(node_features, relations, actions, return_all_q=True)
    assert tuple(q_taken.shape) == (5, 2)
    assert tuple(all_q.shape) == (5, 2, 6)


def test_dense_rgcn_matches_degree_normalized_relation_equation() -> None:
    layer = DenseRGCNLayer(1, 1, 1)
    with torch.no_grad():
        layer.self_linear.weight.fill_(2.0)
        layer.relation_linears[0].weight.fill_(3.0)
        layer.bias.zero_()
    nodes = torch.tensor([[[1.0], [3.0], [5.0]]])
    relations = torch.tensor([[[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    assert torch.equal(layer(nodes, relations), torch.tensor([[[14.0], [9.0], [10.0]]]))


def test_marc_agent_soft_update_runs() -> None:
    agent = MARCAgent(obs_dim=12, action_dim=6, hidden_dim=32)
    agent.soft_update(0.01)


def test_marc_critic_q_taken_matches_all_q_at_chosen_actions() -> None:
    torch.manual_seed(0)
    critic = MARCRelationalCritic(
        n_agents=2,
        node_feature_dim=6,
        action_dim=6,
        hidden_dim=32,
        embed_dim=32,
        num_relations=6,
        num_relational_layers=2,
    )
    node_features = torch.randn(5, 2, 4, 6)
    relations = torch.randint(0, 2, (5, 6, 4, 4), dtype=torch.float32)
    action_idx = torch.randint(0, 6, (5, 2))
    actions = torch.nn.functional.one_hot(action_idx, num_classes=6).to(dtype=torch.float32)

    q_taken, all_q = critic(node_features, relations, actions, return_all_q=True)

    # q_taken must be all_q read out at the chosen actions — the two outputs come from the
    # same per-agent heads, so any drift means the gather indexing is broken.
    gathered = all_q.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(q_taken, gathered)


def test_macpp_adapter_shapes() -> None:
    if not macpp_available():
        pytest.skip("macpp not installed")
    env = MACPPEnv(grid_size=5, n_agents=2, n_pickers=1, n_objects=1, horizon=5, seed=3)
    obs, info = env.reset(seed=3)
    graph_obs = env.graph_observation()
    assert obs.shape == (2, env.obs_dim)
    assert graph_obs.node_features.shape == (2, env.n_entities, env.node_feature_dim)
    assert graph_obs.relations.shape == (env.num_relations, env.n_entities, env.n_entities)
    assert "mean_distance" in info
    env.close()


def test_marc_train_smoke(tmp_path) -> None:
    if not macpp_available():
        pytest.skip("macpp not installed")
    checkpoint = tmp_path / "marc.pt"
    summary = train(
        n_agents=2,
        horizon=6,
        episodes=2,
        seed=5,
        grid_size=5,
        n_pickers=1,
        n_objects=1,
        env_version="v0",
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        embed_dim=32,
        relational_layers=1,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "marc"
    assert summary["env"] == "macpp"
    assert checkpoint.exists()


def _pyg_rgcn_conv(
    layer: DenseRGCNLayer,
    node_features: torch.Tensor,
    relations: torch.Tensor,
) -> torch.Tensor:
    """Transcription of PyG ``RGCNConv``'s reference path (``aggr='mean'``).

    x'_v = Theta_root . x_v + sum_r mean_{u in N_r(v)} (Theta_r . x_u) + b

    Written edge-by-edge with an explicit scatter-mean so that the degree
    normalisation, the isolated-node case and the root/bias terms are all exercised
    independently of ``DenseRGCNLayer``'s dense matmul formulation.
    """
    batch, n_nodes, _ = node_features.shape
    out_dim = layer.self_linear.out_features
    out = torch.zeros(batch, n_nodes, out_dim, dtype=node_features.dtype)
    for b in range(batch):
        for v in range(n_nodes):
            acc = layer.self_linear(node_features[b, v])
            for r, linear in enumerate(layer.relation_linears):
                neighbours = [u for u in range(n_nodes) if relations[b, r, v, u] != 0]
                if not neighbours:
                    continue          # scatter-mean leaves isolated nodes at zero
                messages = torch.stack([linear(node_features[b, u]) for u in neighbours])
                acc = acc + messages.mean(dim=0)
            out[b, v] = acc + layer.bias
    return out


def test_dense_rgcn_layer_matches_the_pyg_rgcn_equation() -> None:
    # The docstring claims DenseRGCNLayer replaces PyG's RGCNConv "without changing its
    # equation". Pin that: root map + per-relation in-degree-normalised mean + shared
    # bias, including the isolated-node branch.
    torch.manual_seed(23)
    batch, n_nodes, in_dim, out_dim, n_relations = 2, 5, 3, 4, 3
    layer = DenseRGCNLayer(in_dim, out_dim, n_relations).double()
    with torch.no_grad():
        layer.bias.copy_(torch.randn(out_dim, dtype=torch.float64))
    node_features = torch.randn(batch, n_nodes, in_dim, dtype=torch.float64)

    relations = (torch.rand(batch, n_relations, n_nodes, n_nodes) < 0.4).double()
    relations[0, 1] = 0.0                     # a relation with no edges at all
    relations[1, 2, 3] = 0.0                  # an isolated node under one relation

    torch.testing.assert_close(
        layer(node_features, relations),
        _pyg_rgcn_conv(layer, node_features, relations),
        rtol=0.0, atol=1e-12,
    )
