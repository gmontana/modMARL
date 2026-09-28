"""CommFormer sparse graph encoder and causal decoder.

Model: a learned receiver-by-sender adjacency supplies both relation embeddings and
attention connectivity to graph Transformer blocks; a MAT-style causal decoder emits
the ordered joint action through per-agent heads.  Invariants: each row contains
exactly ``max(int(n_agents * sparsity), 1)`` learned links at execution, agent ``i``
cannot read later actions, and the architecture gradient reaches ``edges`` only
through the post-softmax adjacency product.  Interface: ``CommFormerBackbone.adjacency``
exposes sampled or exact graphs, while ``act`` and ``forward`` implement autoregressive
rollout and parallel teacher forcing.
Why: the communication graph is the algorithm, so its axes, sparsity, masking, and
gradient path are visible rather than delegated to a third-party GNN package.

Primary paper: Hu et al., "Learning Multi-Agent Communication from Graph Modeling
Perspective," ICLR 2024 (arXiv:2405.08550).  Architecture and training details follow
``charleshsc/CommFormer`` revision ``c6cd65ea0b902703284cb031b7df732c0ff8efa5``.  The
extended arXiv:2411.00382 is consulted only for additional context, not for its
temporal gating extension.

This implements the released ``commformer_dec`` configuration, which every published
script uses (``train_smac.sh``, ``train_smac_comm.sh``, ``train_pp.sh``, ``train_pcp.sh``
and ``train_football.sh`` all set ``algo="commformer_dec"``, and ``train_smac.py`` turns
``dec_actor`` on for any algorithm name containing ``dec``).  Consequently the adjacency
masks attention and the action head is per-agent, matching the reported results rather
than the parser defaults.  Four released quirks are reproduced deliberately and are each
marked at their site: the relation embedding for the (receiver, sender) score is read
transposed; the relation branch is scaled twice, giving it temperature ``1 / head_dim``;
``self_loop_add`` leaves ``2`` on the diagonal so self-attention is doubled after the
softmax; and warmup passes the raw ``edges`` parameter rather than an all-ones graph.
Validation uses the dependency-free three-agent cooperative navigation task.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mat import MATBatch, _orthogonal, _SequencePPOAgent


def _normal_linear(input_dim: int, output_dim: int) -> nn.Linear:
    linear = nn.Linear(input_dim, output_dim)
    nn.init.normal_(linear.weight, std=0.02)
    nn.init.zeros_(linear.bias)
    return linear


def gumbel_topk(
    logits: Tensor,
    topk: int,
    *,
    temperature: float = 1.0,
    hard: bool = True,
) -> Tensor:
    """Released k-hot Gumbel relaxation with a straight-through hard sample."""
    if not 1 <= topk <= logits.shape[-1]:
        raise ValueError("topk must be between one and the row width")
    gumbels = -torch.empty_like(logits).exponential_().log()
    soft = ((logits + gumbels) / temperature).softmax(dim=-1)
    if not hard:
        return soft
    indices = soft.topk(topk, dim=-1).indices
    discrete = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
    return discrete - soft.detach() + soft


class RelationAttention(nn.Module):
    """Relation-enhanced graph attention from CommFormer equations 2 and 3."""

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_agents: int,
        *,
        causal: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = _normal_linear(embedding_dim, 3 * embedding_dim)
        self.relation_projection = _normal_linear(embedding_dim, 2 * embedding_dim)
        self.output = _normal_linear(embedding_dim, embedding_dim)
        self.register_buffer("causal_mask", torch.tril(torch.ones(n_agents, n_agents, dtype=torch.bool)))

    def _project(
        self, query: Tensor, key: Tensor, value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        width = self.embedding_dim
        q = F.linear(query, self.qkv.weight[:width], self.qkv.bias[:width])
        k = F.linear(key, self.qkv.weight[width : 2 * width], self.qkv.bias[width : 2 * width])
        v = F.linear(value, self.qkv.weight[2 * width :], self.qkv.bias[2 * width :])
        batch, target_length = q.shape[:2]
        source_length = k.shape[1]
        head_width = width // self.n_heads
        q = q.view(batch, target_length, self.n_heads, head_width).transpose(1, 2)
        k = k.view(batch, source_length, self.n_heads, head_width).transpose(1, 2)
        v = v.view(batch, source_length, self.n_heads, head_width).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        relation: Tensor | None,
        adjacency: Tensor,
    ) -> Tensor:
        """Attend from receiver rows to sender columns with shape ``(B,N,N)``."""
        q, k, v = self._project(query, key, value)
        batch, _, target_length, head_width = q.shape
        source_length = k.shape[2]
        if relation is None:
            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(head_width)
        else:
            # Released quirk: an upstream ``.transpose(0, 1)`` means the score for
            # receiver i and sender j reads the relation of A[j, i], even though the
            # mask below reads A[i, j].
            relation = relation.transpose(1, 2)
            relation_q, relation_k = self.relation_projection(relation).chunk(2, dim=-1)
            relation_q = relation_q.view(
                batch, target_length, source_length, self.n_heads, head_width,
            ).permute(0, 3, 1, 2, 4)
            relation_k = relation_k.view(
                batch, target_length, source_length, self.n_heads, head_width,
            ).permute(0, 3, 1, 2, 4)
            # Released quirk: this branch is scaled twice (``q *= head_dim ** -0.5``
            # and again by ``1 / sqrt(head_dim)``), so its temperature is 1 / head_dim
            # while the branch above keeps 1 / sqrt(head_dim).
            scores = (
                (q.unsqueeze(3) + relation_q) * (k.unsqueeze(2) + relation_k)
            ).sum(-1) / head_width

        allowed = adjacency[:, None, :target_length, :source_length]
        blocked = allowed == 0
        if self.causal:
            blocked = blocked | ~self.causal_mask[:target_length, :source_length]
        weights = scores.masked_fill(blocked, -torch.inf).softmax(dim=-1)
        # The post-softmax product is what carries the architecture gradient into the
        # adjacency, and it applies the released self-loop doubling.  Causally blocked
        # entries are already zero, so the product need not repeat the causal mask.
        weights = weights * allowed
        attended = torch.einsum("bhts,bhsd->bhtd", weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch, target_length, self.embedding_dim,
        )
        attended = F.dropout(attended, p=self.dropout, training=self.training)
        return self.output(attended)


class _GraphBlock(nn.Module):
    """Released ``GraphTransformerLayer``: graph attention plus a feed-forward."""

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_agents: int,
        *,
        causal: bool,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.attention = RelationAttention(
            embedding_dim, n_heads, n_agents, causal=causal, dropout=dropout,
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_1 = _normal_linear(embedding_dim, embedding_dim)
        self.feed_forward_2 = _normal_linear(embedding_dim, embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, inputs: Tensor, relation: Tensor | None, adjacency: Tensor) -> Tensor:
        attended = self.attention(inputs, inputs, inputs, relation, adjacency)
        hidden = self.attention_norm(
            inputs + F.dropout(attended, p=self.dropout, training=self.training)
        )
        feed_forward = self.feed_forward_2(
            F.dropout(
                F.relu(self.feed_forward_1(hidden)), p=self.dropout, training=self.training,
            )
        )
        return self.output_norm(
            hidden + F.dropout(feed_forward, p=self.dropout, training=self.training)
        )


class _GraphDecoderBlock(nn.Module):
    """Released ``DecodeBlock``: masked self-attention, cross-attention, feed-forward."""

    def __init__(self, embedding_dim: int, n_heads: int, n_agents: int) -> None:
        super().__init__()
        self.action_attention = RelationAttention(
            embedding_dim, n_heads, n_agents, causal=True,
        )
        self.action_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = RelationAttention(
            embedding_dim, n_heads, n_agents, causal=True,
        )
        self.cross_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
            nn.GELU(),
            _orthogonal(nn.Linear(embedding_dim, embedding_dim)),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        actions: Tensor,
        observation_representation: Tensor,
        relation: Tensor | None,
        adjacency: Tensor,
    ) -> Tensor:
        hidden = self.action_norm(
            actions + self.action_attention(actions, actions, actions, relation, adjacency)
        )
        cross = self.cross_attention(
            observation_representation, hidden, hidden, relation, adjacency,
        )
        hidden = self.cross_norm(observation_representation + cross)
        return self.output_norm(hidden + self.feed_forward(hidden))


class CommFormerBackbone(nn.Module):
    """Sparse communication graph, graph encoder, and causal graph decoder."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        *,
        state_dim: int | None = None,
        encode_state: bool = False,
        embedding_dim: int = 64,
        n_heads: int = 1,
        n_blocks: int = 1,
        sparsity: float = 0.4,
        warmup_updates: int = 10,
        post_stable: bool = False,
        post_ratio: float = 0.5,
        relation_enhanced: bool = True,
        share_actor: bool = False,
        self_loop_add: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < sparsity <= 1.0:
            raise ValueError("sparsity must lie in (0, 1]")
        if encode_state and state_dim is None:
            raise ValueError("state_dim is required when encode_state=True")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.encode_state = encode_state
        self.topk = max(int(n_agents * sparsity), 1)
        self.sparsity = sparsity
        self.warmup_updates = warmup_updates
        self.post_stable = post_stable
        self.post_ratio = post_ratio
        self.relation_enhanced = relation_enhanced
        self.share_actor = share_actor
        self.self_loop_add = self_loop_add

        input_dim = state_dim if encode_state else obs_dim
        assert input_dim is not None
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            _orthogonal(nn.Linear(input_dim, embedding_dim), activation=True),
            nn.GELU(),
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.encoder_blocks = nn.ModuleList(
            [
                _GraphBlock(embedding_dim, n_heads, n_agents, causal=False)
                for _ in range(n_blocks)
            ]
        )
        self.value_head = nn.Sequential(
            _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            _orthogonal(nn.Linear(embedding_dim, 1)),
        )
        self.action_encoder = nn.Sequential(
            _orthogonal(nn.Linear(action_dim + 1, embedding_dim, bias=False), activation=True),
            nn.GELU(),
        )
        self.decoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder_blocks = nn.ModuleList(
            [_GraphDecoderBlock(embedding_dim, n_heads, n_agents) for _ in range(n_blocks)]
        )
        self.action_head = self._build_action_head(embedding_dim, action_dim)
        self.edges = nn.Parameter(torch.ones(n_agents, n_agents))
        self.edge_embeddings = nn.Embedding(2, embedding_dim)
        self.register_buffer("last_adjacency", torch.ones(n_agents, n_agents))

    def _build_action_head(self, embedding_dim: int, action_dim: int) -> nn.Module:
        """Released ``dec_actor`` head: one shared actor or one actor per agent."""
        if self.share_actor:
            return nn.Sequential(
                nn.LayerNorm(embedding_dim),
                _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                _orthogonal(nn.Linear(embedding_dim, action_dim)),
            )
        return nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                _orthogonal(nn.Linear(embedding_dim, action_dim)),
            )
            for _ in range(self.n_agents)
        )

    def model_parameters(self) -> list[nn.Parameter]:
        """All architectural weights except the upper-level graph parameter."""
        return [parameter for name, parameter in self.named_parameters() if name != "edges"]

    def edge_parameters(self) -> list[nn.Parameter]:
        """The sole upper-level adjacency-logit parameter."""
        return [self.edges]

    def _exact_graph(self) -> Tensor:
        """Deterministic straight-through top-k graph used at execution."""
        soft = self.edges.softmax(dim=-1)
        indices = self.edges.topk(self.topk, dim=-1).indices
        hard = torch.zeros_like(self.edges).scatter_(-1, indices, 1.0)
        return hard - soft.detach() + soft

    def _record(self, graph: Tensor) -> Tensor:
        self.last_adjacency.copy_(graph.detach())
        return graph

    def adjacency(
        self,
        *,
        exact: bool,
        training_step: int = 0,
        total_steps: int = 0,
    ) -> Tensor:
        """Return warmup, sampled k-hot, or deterministic top-k connectivity."""
        if exact:
            return self._record(self._exact_graph())
        if training_step <= self.warmup_updates:
            # Released: warmup passes the raw parameter rather than an all-ones graph.
            # The edge optimizer already steps during warmup, so the two differ.
            graph = self.edges
        else:
            graph = gumbel_topk(self.edges, self.topk)
        if self.post_stable and training_step > int(self.post_ratio * total_steps):
            graph = self._exact_graph()
        return self._record(graph)

    def _connectivity(self, graph: Tensor) -> Tensor:
        """Apply the released self-loop to the attention mask only."""
        identity = torch.eye(self.n_agents, dtype=graph.dtype, device=graph.device)
        if self.self_loop_add:
            # Released default: the diagonal becomes A_ii + 1, so the post-softmax
            # product in RelationAttention doubles each agent's self-attention.
            return graph + identity
        return graph * (1.0 - identity) + identity

    def _relations(self, graph: Tensor, batch: int) -> Tensor | None:
        if not self.relation_enhanced:
            return None
        # Released: the embedding indexes the raw graph, before the self-loop, and the
        # integer cast keeps the architecture gradient out of this path entirely.
        embedded = self.edge_embeddings(graph.long())
        return embedded.unsqueeze(0).expand(batch, -1, -1, -1)

    def _graph_inputs(
        self,
        batch: int,
        *,
        exact: bool,
        training_step: int,
        total_steps: int,
    ) -> tuple[Tensor, Tensor | None]:
        graph = self.adjacency(
            exact=exact, training_step=training_step, total_steps=total_steps,
        )
        mask = self._connectivity(graph).unsqueeze(0).expand(batch, -1, -1)
        return mask, self._relations(graph, batch)

    def encode(
        self,
        obs: Tensor,
        state: Tensor | None = None,
        *,
        exact: bool,
        training_step: int = 0,
        total_steps: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Encode observations and return values, representations, graph, and relations."""
        if obs.ndim != 3 or obs.shape[1:] != (self.n_agents, self.obs_dim):
            raise ValueError("obs must match (batch, agents, obs_dim)")
        if self.encode_state:
            if state is None or state.shape[:2] != obs.shape[:2] or state.shape[-1] != self.state_dim:
                raise ValueError("state must match (batch, agents, state_dim)")
            inputs = state
        else:
            inputs = obs
        adjacency, relation = self._graph_inputs(
            obs.shape[0], exact=exact, training_step=training_step, total_steps=total_steps,
        )
        representation = self.encoder_norm(self.input_encoder(inputs))
        for block in self.encoder_blocks:
            representation = block(representation, relation, adjacency)
        return self.value_head(representation).squeeze(-1), representation, adjacency, relation

    def shifted_actions(self, actions: Tensor) -> Tensor:
        shifted = torch.zeros(
            actions.shape[0], self.n_agents, self.action_dim + 1,
            dtype=torch.float32, device=actions.device,
        )
        shifted[:, 0, 0] = 1.0
        if self.n_agents > 1:
            shifted[:, 1:, 1:] = F.one_hot(
                actions[:, :-1].long(), self.action_dim,
            ).to(shifted.dtype)
        return shifted

    def decode(
        self,
        shifted_actions: Tensor,
        representation: Tensor,
        adjacency: Tensor,
        relation: Tensor | None,
    ) -> Tensor:
        hidden = self.decoder_norm(self.action_encoder(shifted_actions))
        for block in self.decoder_blocks:
            hidden = block(hidden, representation, relation, adjacency)
        if self.share_actor:
            return self.action_head(hidden)
        return torch.stack(
            [head(hidden[:, agent]) for agent, head in enumerate(self.action_head)], dim=1,
        )

    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
        available_actions: Tensor | None = None,
        state: Tensor | None = None,
        *,
        training_step: int = 0,
        total_steps: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Teacher-force actions through a sampled training communication graph."""
        values, representation, adjacency, relation = self.encode(
            obs,
            state,
            exact=False,
            training_step=training_step,
            total_steps=total_steps,
        )
        logits = self.decode(self.shifted_actions(actions), representation, adjacency, relation)
        if available_actions is not None:
            logits = logits.masked_fill(~available_actions.bool(), torch.finfo(logits.dtype).min)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(actions.long()), values, distribution.entropy()

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        available_actions: Tensor | None = None,
        *,
        state: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Autoregressively act with the exact top-k execution graph."""
        values, representation, adjacency, relation = self.encode(obs, state, exact=True)
        batch = obs.shape[0]
        shifted = torch.zeros(
            batch, self.n_agents, self.action_dim + 1, dtype=obs.dtype, device=obs.device,
        )
        shifted[:, 0, 0] = 1.0
        actions = torch.zeros(batch, self.n_agents, dtype=torch.long, device=obs.device)
        log_probs = torch.zeros(batch, self.n_agents, dtype=obs.dtype, device=obs.device)
        for agent in range(self.n_agents):
            logits = self.decode(shifted, representation, adjacency, relation)[:, agent]
            if available_actions is not None:
                logits = logits.masked_fill(
                    ~available_actions[:, agent].bool(), torch.finfo(logits.dtype).min,
                )
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            actions[:, agent] = action
            log_probs[:, agent] = distribution.log_prob(action)
            if agent + 1 < self.n_agents:
                shifted[:, agent + 1, 1:] = F.one_hot(
                    action, self.action_dim,
                ).to(shifted.dtype)
        return actions, log_probs, values

    def values(self, obs: Tensor, state: Tensor | None = None) -> Tensor:
        """Return normalized values under the exact execution graph."""
        return self.encode(obs, state, exact=True)[0]

    @torch.no_grad()
    def communication_rate(self) -> float:
        """Fraction of possible non-self directed links in the exact graph."""
        graph = self.adjacency(exact=True).detach()
        off_diagonal = ~torch.eye(self.n_agents, dtype=torch.bool, device=graph.device)
        return float(graph[off_diagonal].mean()) if self.n_agents > 1 else 0.0


class CommFormerAgent(_SequencePPOAgent):
    """Complete discrete CommFormer learner with sparse graph optimization."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        *,
        state_dim: int | None = None,
        encode_state: bool = False,
        embedding_dim: int = 64,
        n_heads: int = 1,
        n_blocks: int = 1,
        sparsity: float = 0.4,
        warmup_updates: int = 10,
        bilevel: bool = True,
        post_stable: bool = False,
        post_ratio: float = 0.5,
        relation_enhanced: bool = True,
        share_actor: bool = False,
        self_loop_add: bool = True,
        learning_rate: float = 5e-4,
        edge_learning_rate: float = 1e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 10.0,
        huber_delta: float = 10.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        backbone = CommFormerBackbone(
            n_agents,
            obs_dim,
            action_dim,
            state_dim=state_dim,
            encode_state=encode_state,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            sparsity=sparsity,
            warmup_updates=warmup_updates,
            post_stable=post_stable,
            post_ratio=post_ratio,
            relation_enhanced=relation_enhanced,
            share_actor=share_actor,
            self_loop_add=self_loop_add,
        )
        super().__init__(
            backbone,
            n_agents=n_agents,
            action_dim=action_dim,
            learning_rate=learning_rate,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            huber_delta=huber_delta,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self.bilevel = bilevel
        self.optimizer = torch.optim.Adam(
            backbone.model_parameters(), lr=learning_rate, eps=1e-5, weight_decay=0.0,
        )
        self.edge_optimizer = torch.optim.Adam(
            backbone.edge_parameters(), lr=edge_learning_rate,
        )

    @property
    def graph(self) -> CommFormerBackbone:
        assert isinstance(self.backbone, CommFormerBackbone)
        return self.backbone

    def _evaluate(
        self, batch: MATBatch, index: Tensor, training_step: int, total_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        states = None if batch.states is None else batch.states[index]
        return self.graph(
            batch.obs[index],
            batch.actions[index],
            batch.available_actions[index],
            states,
            training_step=training_step,
            total_steps=total_steps,
        )

    def _edge_epoch(self, epoch: int, training_step: int, total_steps: int) -> bool:
        if not self.bilevel or (epoch + 1) % 5:
            return False
        if not self.graph.post_stable:
            return True
        return total_steps <= 0 or training_step <= int(self.graph.post_ratio * total_steps)

    def _optimize(self, loss: Tensor, epoch: int, training_step: int, total_steps: int) -> str:
        self.optimizer.zero_grad(set_to_none=True)
        self.edge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.graph.model_parameters(), self.max_grad_norm)
        if self._edge_epoch(epoch, training_step, total_steps):
            self.edge_optimizer.step()
            return "edge"
        self.optimizer.step()
        return "model"

    def communication_rate(self) -> float:
        """Return exact non-self directed link utilization in ``[0, 1]``."""
        return self.graph.communication_rate()

__all__ = ["CommFormerAgent", "CommFormerBackbone", "RelationAttention", "gumbel_topk"]
