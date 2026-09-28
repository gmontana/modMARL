"""MARC implementation.

Original paper:
Sharlin Utke, Jeremie Houssineau, and Giovanni Montana. "Investigating
Relational State Abstraction in Collaborative MARL." arXiv:2412.15388, 2024.

Official code: https://github.com/sharlinu/MARC, reference commit
43b71357bce11b075e799876e8c4f1deade787d8.

MARC uses decentralized three-layer policies with non-affine input BatchNorm and
an algorithm-specific centralized relational critic. Each agent's observation is
represented as a typed graph, processed by the released R-GCN equation (root map
plus relation-specific degree-normalised aggregation), globally max pooled, and
combined with every other agent's executed action. Each agent owns a distinct
critic head. No information is exchanged at execution.

The dependency-free dense layer replaces PyG's ``RGCNConv`` without changing its
equation. The entropy coefficient is ``1 / 100 = 0.01``, matching the released
``reward_scale=100`` and paper Table 2; note the paper's prose instead says 0.05, which
contradicts its own table.

A single relational layer is used, as the paper specifies ("a single-layered, shared
R-GCN module") and as every published pick-and-place config selects (``net_code:
1g1i1f``). The release's two-layer ablation *ties* its layers -- it appends one
``RGCNConv`` instance twice into a PyG ``Sequential``, which stores it by ``setattr``
without copying -- so stacking independently-parameterised layers has no counterpart
upstream.

Deliberate divergence from the release: ``update_critic`` there sets
``requires_grad = False`` on every ``gnn_layers`` parameter of both the critic and the
target critic on every update, and ``update_policies`` disables critic gradients, so the
released R-GCN stays at its initialization for the whole run and only the entity encoder
and the per-agent heads learn. modMARL trains the relational trunk, implementing the
learned relational abstraction the paper describes; the freeze is unremarked in the
paper and appears to be an oversight.
"""

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..components import soft_update_module


class DenseRGCNLayer(nn.Module):
    """Dense relation-aware message passing layer for small MARL graphs."""

    def __init__(self, input_dim: int, output_dim: int, num_relations: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.relation_linears = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_relations)]
        )
        self.bias = nn.Parameter(Tensor(output_dim))
        nn.init.zeros_(self.bias)

    def forward(self, node_features: Tensor, relations: Tensor) -> Tensor:
        if relations.ndim != 4:
            raise ValueError(
                "expected relations with shape [batch, num_relations, num_nodes, num_nodes], "
                f"got {tuple(relations.shape)}"
            )
        if node_features.ndim != 3:
            raise ValueError(
                "expected node_features with shape [batch, num_nodes, feature_dim], "
                f"got {tuple(node_features.shape)}"
            )

        # Convention: relations[:, r][v, u] == 1 means "u is <r> of v", so row v lists
        # node v's relation-r neighbours and messages[v] is their normalised mean.
        updated = self.self_linear(node_features)
        for relation_id, linear in enumerate(self.relation_linears):
            adjacency = relations[:, relation_id]
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            messages = adjacency.matmul(node_features) / degree
            updated = updated + linear(messages)
        return updated + self.bias





def build_leaky_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(input_dim, affine=False),
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class MARCActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = build_leaky_mlp(obs_dim, hidden_dim, action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)

    def sample(
        self,
        obs: Tensor,
        temperature: float = 1.0,
        hard: bool = True,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits = self(obs)
        del temperature, hard
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_idx = probs.argmax(dim=-1) if deterministic else torch.distributions.Categorical(probs=probs).sample()
        one_hot = F.one_hot(action_idx, num_classes=self.action_dim).to(dtype=logits.dtype)
        chosen_log_prob = log_probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return one_hot, action_idx, logits, probs, log_probs, chosen_log_prob, entropy


class MARCAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = MARCActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.target_actor = copy.deepcopy(self.actor)

    def soft_update(self, tau: float) -> None:
        soft_update_module(self.target_actor, self.actor, tau)


class MARCRelationalCritic(nn.Module):
    """Shared relational critic over per-agent graphs and joint actions."""

    def __init__(
        self,
        n_agents: int,
        node_feature_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_relations: int = 6,
        num_relational_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.node_encoder = nn.Linear(node_feature_dim, embed_dim)
        self.relational_layers = nn.ModuleList(
            [DenseRGCNLayer(embed_dim, embed_dim, num_relations) for _ in range(num_relational_layers)]
        )
        head_input_dim = embed_dim + action_dim * max(0, n_agents - 1)
        self.agent_q_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_input_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(n_agents)
            ]
        )

    def shared_parameters(self):
        return chain(self.node_encoder.parameters(), self.relational_layers.parameters())

    def scale_shared_grads(self) -> None:
        for parameter in self.shared_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(1.0 / self.n_agents)

    def encode_graphs(self, node_features: Tensor, relations: Tensor) -> Tensor:
        encoded = self.node_encoder(node_features)
        for layer in self.relational_layers:
            encoded = F.relu(layer(encoded, relations))
        return encoded.max(dim=1).values

    def forward(
        self,
        node_features: Tensor,
        relations: Tensor,
        actions: Tensor,
        return_all_q: bool = True,
    ) -> tuple[Tensor, Tensor]:
        graph_embeddings = [
            self.encode_graphs(node_features[:, agent_id], relations)
            for agent_id in range(self.n_agents)
        ]

        all_q_values = []
        chosen_q_values = []
        for agent_id in range(self.n_agents):
            other_actions = [
                actions[:, other_id]
                for other_id in range(self.n_agents)
                if other_id != agent_id
            ]
            critic_input = torch.cat([graph_embeddings[agent_id], *other_actions], dim=-1)
            q_values = self.agent_q_heads[agent_id](critic_input)
            all_q_values.append(q_values)
            chosen_idx = actions[:, agent_id].argmax(dim=-1, keepdim=True)
            chosen_q_values.append(q_values.gather(1, chosen_idx).squeeze(-1))

        q_taken = torch.stack(chosen_q_values, dim=1)
        all_q = torch.stack(all_q_values, dim=1)
        if not return_all_q:
            all_q = torch.empty(0, device=node_features.device)
        return q_taken, all_q




__all__ = [
    "MARCActor",
    "MARCAgent",
    "MARCRelationalCritic",
]
