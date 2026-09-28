"""Golden graph, gradient, causality, and bi-level contracts for CommFormer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modmarl.algorithms.commformer import (
    CommFormerAgent,
    CommFormerBackbone,
    RelationAttention,
    gumbel_topk,
)
from modmarl.algorithms.mat import MATBatch


def _batch(learner: CommFormerAgent, samples: int = 3) -> MATBatch:
    obs = torch.randn(samples, learner.n_agents, learner.graph.obs_dim)
    with torch.no_grad():
        actions, log_probs, values = learner.act(obs)
    return MATBatch(
        obs=obs,
        actions=actions,
        old_log_probs=log_probs,
        old_values=values,
        advantages=torch.randn_like(values),
        returns=torch.randn_like(values),
        active_masks=torch.ones_like(values),
        available_actions=torch.ones(
            samples, learner.n_agents, learner.action_dim, dtype=torch.bool,
        ),
    )


def test_gumbel_topk_is_k_hot_and_straight_through():
    torch.manual_seed(3)
    logits = torch.zeros(4, 5, requires_grad=True)
    graph = gumbel_topk(logits, 2)
    torch.testing.assert_close(graph.sum(-1), torch.full((4,), 2.0))
    assert set(graph.detach().unique().tolist()) == {0.0, 1.0}
    (graph * torch.arange(5.0)).sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_exact_graph_uses_receiver_rows_and_sender_columns():
    model = CommFormerBackbone(3, 4, 3, embedding_dim=8, sparsity=1 / 3)
    with torch.no_grad():
        model.edges.copy_(torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 2.0], [1.0, 2.0, 5.0]]))
    expected = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(model.adjacency(exact=True).detach(), expected)


def test_warmup_uses_the_raw_edge_parameter_then_the_graph_is_sparse():
    torch.manual_seed(5)
    model = CommFormerBackbone(
        5, 3, 4, embedding_dim=10, sparsity=0.4, warmup_updates=10,
    )
    # Released behaviour: warmup returns `edges` itself, which only equals a full graph
    # while the parameter still holds its all-ones initialization.
    warm = model.adjacency(exact=False, training_step=10)
    assert warm is model.edges
    torch.testing.assert_close(warm.detach(), torch.ones(5, 5))
    sparse = model.adjacency(exact=False, training_step=11)
    torch.testing.assert_close(sparse.sum(-1), torch.full((5,), 2.0))


def test_graph_mask_blocks_disconnected_sender_information():
    torch.manual_seed(7)
    attention = RelationAttention(4, 1, 3, causal=False).eval()
    query = torch.randn(1, 3, 4)
    key = torch.randn(1, 3, 4)
    value = torch.randn(1, 3, 4)
    adjacency = torch.eye(3).unsqueeze(0)
    first = attention(query, key, value, None, adjacency)
    changed = value.clone()
    changed[:, 1] += 100.0
    second = attention(query, key, changed, None, adjacency)
    torch.testing.assert_close(first[:, 0], second[:, 0])


def test_adjacency_gradient_flows_through_the_post_softmax_product():
    torch.manual_seed(11)
    model = CommFormerBackbone(
        3, 4, 3, embedding_dim=8, sparsity=2 / 3, warmup_updates=-1,
    ).eval()
    obs = torch.randn(2, 3, 4)
    actions = torch.tensor([[0, 1, 2], [2, 1, 0]])
    log_probs, values, _ = model(obs, actions, training_step=20, total_steps=100)
    (log_probs.sum() + values.sum()).backward()
    assert model.edges.grad is not None
    assert model.edges.grad.abs().sum() > 0


def test_commformer_teacher_forcing_matches_autoregressive_evaluation():
    torch.manual_seed(13)
    model = CommFormerBackbone(
        3, 4, 3, embedding_dim=8, sparsity=1.0, warmup_updates=-1,
    ).eval()
    obs = torch.randn(2, 3, 4)
    actions, sampled, sampled_values = model.act(obs)
    parallel, parallel_values, _ = model(obs, actions, training_step=20)
    torch.testing.assert_close(sampled, parallel)
    torch.testing.assert_close(sampled_values, parallel_values)


def test_model_and_edge_optimizers_have_disjoint_ownership():
    learner = CommFormerAgent(3, 4, 3, embedding_dim=8)
    model_parameters = {
        id(parameter) for group in learner.optimizer.param_groups for parameter in group["params"]
    }
    edge_parameters = {
        id(parameter) for group in learner.edge_optimizer.param_groups for parameter in group["params"]
    }
    assert model_parameters.isdisjoint(edge_parameters)
    assert edge_parameters == {id(learner.graph.edges)}


def test_bilevel_schedule_updates_edges_every_fifth_epoch():
    torch.manual_seed(17)
    learner = CommFormerAgent(
        3, 4, 3, embedding_dim=8, sparsity=2 / 3, warmup_updates=-1,
    )
    metrics = learner.update(_batch(learner), epochs=10, training_step=20, total_steps=100)
    assert metrics["model_updates"] == 8
    assert metrics["edge_updates"] == 2


def test_bilevel_steps_change_only_the_selected_parameter_owner():
    learner = CommFormerAgent(
        3, 4, 3, embedding_dim=8, sparsity=2 / 3, warmup_updates=-1,
    )
    model_parameter = next(iter(learner.graph.model_parameters()))
    model_before = model_parameter.detach().clone()
    edge_before = learner.graph.edges.detach().clone()
    learner._optimize(model_parameter.sum() + learner.graph.edges.square().sum(), 0, 20, 100)
    assert not torch.equal(model_parameter, model_before)
    torch.testing.assert_close(learner.graph.edges, edge_before)

    model_before = model_parameter.detach().clone()
    edge_before = learner.graph.edges.detach().clone()
    learner._optimize(model_parameter.sum() + learner.graph.edges.square().sum(), 4, 20, 100)
    torch.testing.assert_close(model_parameter, model_before)
    assert not torch.equal(learner.graph.edges, edge_before)


def test_post_stable_stops_upper_level_updates_after_ratio():
    torch.manual_seed(19)
    learner = CommFormerAgent(
        3,
        4,
        3,
        embedding_dim=8,
        sparsity=2 / 3,
        warmup_updates=-1,
        post_stable=True,
        post_ratio=0.5,
    )
    metrics = learner.update(_batch(learner), epochs=5, training_step=60, total_steps=100)
    assert metrics["model_updates"] == 5
    assert metrics["edge_updates"] == 0


def test_release_settings_and_exact_communication_rate():
    learner = CommFormerAgent(5, 4, 3)
    assert learner.graph.topk == 2
    assert learner.graph.warmup_updates == 10
    assert learner.bilevel
    # The published scripts all run `commformer_dec`: per-agent heads, added self loop.
    assert not learner.graph.share_actor
    assert learner.graph.self_loop_add
    assert len(learner.graph.action_head) == 5
    assert learner.optimizer.param_groups[0]["lr"] == 5e-4
    assert learner.edge_optimizer.param_groups[0]["lr"] == 1e-4
    with torch.no_grad():
        learner.graph.edges.copy_(torch.eye(5) * 10.0)
    assert learner.communication_rate() == 0.25


def test_checkpoint_round_trip_preserves_graph_and_deterministic_policy(tmp_path):
    torch.manual_seed(29)
    learner = CommFormerAgent(3, 4, 3, embedding_dim=8, sparsity=2 / 3).eval()
    with torch.no_grad():
        learner.graph.edges.copy_(torch.randn_like(learner.graph.edges))
    obs = torch.randn(1, 3, 4)
    expected = learner.act(obs, deterministic=True)[0]
    checkpoint = tmp_path / "commformer.pt"
    torch.save(learner.state_dict(), checkpoint)
    restored = CommFormerAgent(3, 4, 3, embedding_dim=8, sparsity=2 / 3).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.graph.edges, learner.graph.edges)
    torch.testing.assert_close(restored.act(obs, deterministic=True)[0], expected)


def test_self_loop_add_doubles_the_diagonal_and_clamping_does_not():
    graph = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    added = CommFormerBackbone(2, 3, 2, embedding_dim=8, self_loop_add=True)
    clamped = CommFormerBackbone(2, 3, 2, embedding_dim=8, self_loop_add=False)
    torch.testing.assert_close(added._connectivity(graph), torch.tensor([[2.0, 0.0], [1.0, 2.0]]))
    torch.testing.assert_close(clamped._connectivity(graph), torch.tensor([[1.0, 0.0], [1.0, 1.0]]))


def test_decoder_head_is_per_agent_unless_the_actor_is_shared():
    torch.manual_seed(31)
    per_agent = CommFormerBackbone(4, 3, 2, embedding_dim=8)
    shared = CommFormerBackbone(4, 3, 2, embedding_dim=8, share_actor=True)
    assert len(per_agent.action_head) == 4
    assert isinstance(shared.action_head, nn.Sequential)

    hidden = torch.randn(2, 4, 8)
    def logits() -> torch.Tensor:
        return torch.stack(
            [head(hidden[:, agent]) for agent, head in enumerate(per_agent.action_head)],
            dim=1,
        )

    before = logits()
    with torch.no_grad():
        per_agent.action_head[2][-1].bias.add_(1.0)
    changed = (logits() - before).abs().amax(dim=(0, 2))
    assert changed[2] > 0
    assert changed[[0, 1, 3]].max() == 0


def _released_relation_attention(
    module: RelationAttention,
    inputs: Tensor,
    relation: Tensor,
    graph: Tensor,
    *,
    self_loop_add: bool,
    masked: bool,
) -> Tensor:
    """Transcription of the released ``RelationMultiheadAttention.forward``.

    ``graph`` is the raw adjacency because the release adds the self loop inside
    attention, where modMARL adds it once in ``_connectivity``.  The time-first layout
    and the HuggingFace ``Conv1D`` weight orientation are reproduced verbatim from
    ``charleshsc/CommFormer@c6cd65e`` so that any drift in our batch-first rewrite --
    including the transposed relation index and the doubly scaled relation branch --
    surfaces here as a numerical mismatch.
    """
    n_agents, embed = inputs.shape[1], module.embedding_dim
    heads = module.n_heads
    head_dim = embed // heads
    query = inputs.permute(1, 0, 2).contiguous()
    relation_tb = relation.permute(1, 2, 0, 3).contiguous()
    attn_mask = graph.permute(1, 2, 0).contiguous()
    length, batch, _ = query.shape

    weight, bias = module.qkv.weight, module.qkv.bias
    q = F.linear(query, weight[:embed], bias[:embed]).view(length, batch * heads, head_dim)
    k = F.linear(query, weight[embed : 2 * embed], bias[embed : 2 * embed]).view(
        length, batch * heads, head_dim,
    )
    v = F.linear(query, weight[2 * embed :], bias[2 * embed :]).view(
        length, batch * heads, head_dim,
    )

    projected = (
        relation_tb @ module.relation_projection.weight.t() + module.relation_projection.bias
    )
    relation_q, relation_k = projected.chunk(2, dim=-1)
    relation_q = relation_q.contiguous().view(
        length, length, batch * heads, head_dim,
    ).transpose(0, 1)
    relation_k = relation_k.contiguous().view(
        length, length, batch * heads, head_dim,
    ).transpose(0, 1)
    q = (q.unsqueeze(1) + relation_q) * head_dim**-0.5
    k = k.unsqueeze(0) + relation_k
    scores = torch.einsum("ijbn,ijbn->ijb", q, k) * (1.0 / math.sqrt(head_dim))

    if masked:
        causal = torch.tril(torch.ones(n_agents, n_agents, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(-1), -torch.inf)
    self_loop = torch.eye(length, dtype=attn_mask.dtype).unsqueeze(-1)
    mask = attn_mask + self_loop if self_loop_add else attn_mask * (1 - self_loop) + self_loop
    weights = scores.masked_fill(mask == 0, -torch.inf).softmax(dim=1) * mask

    attended = torch.einsum("ijb,jbn->bin", weights, v)
    attended = attended.transpose(0, 1).contiguous().view(length, batch, embed)
    attended = attended @ module.output.weight.t() + module.output.bias
    return attended.permute(1, 0, 2).contiguous()


def test_relation_attention_matches_the_released_formulation():
    torch.manual_seed(41)
    n_agents, embed, batch = 4, 8, 2
    graph = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0],
         [0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 1.0, 1.0],
         [1.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    table = torch.randn(2, embed, dtype=torch.float64)
    relation = table[graph.long()].unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    raw = graph.unsqueeze(0).expand(batch, -1, -1).contiguous()
    inputs = torch.randn(batch, n_agents, embed, dtype=torch.float64)
    identity = torch.eye(n_agents, dtype=torch.float64)

    for causal in (False, True):
        for self_loop_add in (True, False):
            module = RelationAttention(embed, 1, n_agents, causal=causal).double().eval()
            connectivity = (
                graph + identity if self_loop_add else graph * (1.0 - identity) + identity
            )
            got = module(
                inputs, inputs, inputs, relation,
                connectivity.unsqueeze(0).expand(batch, -1, -1),
            )
            expected = _released_relation_attention(
                module, inputs, relation, raw,
                self_loop_add=self_loop_add, masked=causal,
            )
            torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-12)
