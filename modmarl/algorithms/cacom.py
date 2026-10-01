"""CACOM: two-stage receiver-contextual communication under a bit budget.

Original paper:
Xinran Li and Jun Zhang. "Context-aware Communication for Multi-agent
Reinforcement Learning." AAMAS 2024. arXiv:2312.15600v3.
Official code: https://github.com/LXXXXR/CACOM (Apache-2.0), reference commit
97493a0b2c402e88a06d4e0d21327c41bbd21709.

Model: each receiver first broadcasts a short quantized request derived from its
entity-token observation and recurrent history. Every potential helper uses that
request as a cross-attention query over its own entity features, gates the directed
link, and returns a quantized receiver-specific response. The receiver jointly
attends to local entities and incoming responses before its recurrent Q head.
Invariants: response axes are always (helper, receiver); self links are absent;
both channel stages use learned-step-size quantization; gate gradients are isolated
from the ordinary TD/auxiliary update. Interface: ``CACOMNetwork.step`` exposes all
communication products, and ``CACOMAgent`` owns online/target networks and QMIX.

Paper/code reconciliation: paper Equation (7) defines gate labels through the
mixed global value. The released ``forward_gate`` instead compares the recipient's
maximum local Q with a sampled helper link forced on and off. ``gate_labels`` follows
the released rule by default for reproducibility. The explicit ``paper`` gate-label
mode instead evaluates the link-on and link-off actions under a common link-on
value function and mixer, holding other agents' actions fixed. This implements
the action counterfactual in Equations (7)--(11); its common communication context
is an explicit resolution of the paper's underspecified critic context. The paper's
implementation appendix specifies Adam for communication learning, but the released
learner uses RMSProp (alpha 0.99, epsilon 1e-5); the trainer follows that release.
The gate scores are scaled by
1/sqrt(d_k) as paper Equation (6) writes them; the release omits that factor, so the
paper governs here. The target network deliberately excludes the
gate, reproducing the release's controller-owned ``ExpGate``. Environments
with an entity schema should pass ``entity_schema`` as ``(count, length)`` pairs per
entity *type*, mirroring the release's ``obs_segs``: every token of a type shares one
encoder, which is what makes the encoder permutation-equivariant over interchangeable
entities. Without an environment schema, each scalar observation feature is conservatively
treated as its own single-token type; validation always supplies the navigation schema.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qmix import QMixer


def _grad_scale(value: Tensor, scale: float) -> Tensor:
    return (value - value * scale).detach() + value * scale


def _round_straight_through(value: Tensor) -> Tensor:
    return (value.round() - value).detach() + value


class LearnedStepQuantizer(nn.Module):
    """Paper Equation (4), using the released symmetric LSQ bounds."""

    def __init__(self, bits: int = 2) -> None:
        super().__init__()
        if bits < 2:
            raise ValueError("CACOM's signed LSQ channel requires at least two bits")
        self.lower = -(2 ** (bits - 1)) + 1
        self.upper = 2 ** (bits - 1) - 1
        self.step_size = nn.Parameter(torch.ones(1))

    def forward(self, value: Tensor) -> Tensor:
        scale = 1.0 / math.sqrt(self.upper * value.numel())
        step = _grad_scale(self.step_size.abs().clamp_min(1e-8), scale)
        codes = _round_straight_through((value / step).clamp(self.lower, self.upper))
        return codes * step


class CACOMNetwork(nn.Module):
    """Paper Sections 4.1--4.3 in one entity-token communication network."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        *,
        entity_schema: tuple[tuple[int, int], ...] | None = None,
        encode_dim: int = 8,
        request_dim: int = 4,
        response_dim: int = 8,
        hidden_dim: int = 64,
        bits: int = 2,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # (count, length) per entity TYPE, mirroring the release's `obs_segs`: one
        # encoder is shared by every token of a type, which is what makes the encoder
        # permutation-equivariant over interchangeable entities.
        self.entity_schema = entity_schema or ((1, 1),) * obs_dim
        if sum(count * length for count, length in self.entity_schema) != obs_dim:
            raise ValueError("entity_schema must partition obs_dim")
        self.n_entities = sum(count for count, _ in self.entity_schema)
        self.encode_dim = encode_dim
        self.request_dim = request_dim
        self.response_dim = response_dim
        self.hidden_dim = hidden_dim

        self.entity_encoders = nn.ModuleList(
            nn.Linear(length, encode_dim) for _, length in self.entity_schema
        )
        self.entity_kqv = nn.Linear(encode_dim, 3 * encode_dim)
        self.entity_ff = nn.Sequential(
            nn.Linear(encode_dim, 4 * encode_dim), nn.LeakyReLU(),
            nn.Linear(4 * encode_dim, encode_dim),
        )
        self.request_head = nn.Linear(hidden_dim + self.n_entities * encode_dim, request_dim)
        self.request_quantizer = LearnedStepQuantizer(bits)

        self.response_key_value = nn.Linear(encode_dim, 2 * encode_dim)
        self.response_query = nn.Linear(request_dim, encode_dim)
        self.response_head = nn.Sequential(
            nn.Linear(encode_dim, 4 * encode_dim), nn.LeakyReLU(),
            nn.Linear(4 * encode_dim, response_dim),
        )
        self.response_quantizer = LearnedStepQuantizer(bits)

        self.gate_key = nn.Linear(encode_dim, encode_dim)
        self.gate_query = nn.Linear(request_dim, encode_dim)
        self.gate_head = nn.Linear(self.n_entities, 2)

        self.policy_feature_kqv = nn.Sequential(nn.LeakyReLU(), nn.Linear(encode_dim, 3 * encode_dim))
        self.policy_message_kqv = nn.Linear(response_dim, 3 * encode_dim)
        token_count = self.n_entities + n_agents - 1
        self.policy_input = nn.Linear(token_count * encode_dim, hidden_dim)
        self.policy_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

        self.predict_feature_kv = nn.Linear(encode_dim, 2 * encode_dim)
        self.predict_message_kqv = nn.Linear(response_dim, 3 * encode_dim)
        self.predict_q = nn.Linear((n_agents - 1) * encode_dim, (n_agents - 1) * action_dim)

    def init_hidden(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.n_agents, self.hidden_dim, device=device)

    def encode(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Return entity features and quantized receiver requests."""
        pieces = torch.split(
            obs, [count * length for count, length in self.entity_schema], dim=-1,
        )
        tokens = torch.cat([
            encoder(piece.unflatten(-1, (count, length)))
            for encoder, piece, (count, length)
            in zip(self.entity_encoders, pieces, self.entity_schema)
        ], dim=-2)
        key, query, value = self.entity_kqv(tokens).chunk(3, dim=-1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.encode_dim**0.5
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        features = tokens + attended
        features = features + self.entity_ff(features)
        requests = self.request_head(torch.cat([hidden, features.flatten(2)], dim=-1))
        return features, self.request_quantizer(requests)

    def personalized_responses(self, features: Tensor, requests: Tensor) -> Tensor:
        """Cross-attended messages with axes ``(B, helper, receiver, D)``."""
        key, value = self.response_key_value(features).chunk(2, dim=-1)
        query = self.response_query(requests)
        scores = torch.einsum("bhme,bre->bhrm", key, query) / self.encode_dim**0.5
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bhrm,bhme->bhre", weights, value)
        return self.response_head(attended)

    def gate_logits(self, features: Tensor, requests: Tensor) -> Tensor:
        """Paper Equation (6) logits with axes ``(B, helper, receiver, 2)``.

        The 1/sqrt(d_k) scaling is the paper's; the release's ``ExpGate`` computes the
        bare ``q @ k``.
        """
        key = self.gate_key(features)
        query = self.gate_query(requests)
        evidence = torch.einsum("bhme,bre->bhrm", key, query) / self.encode_dim**0.5
        return self.gate_head(evidence)

    def communication(
        self,
        features: Tensor,
        requests: Tensor,
        *,
        force_all_links: bool = False,
        forced_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate, gate, quantize, and receiver-order personalised responses."""
        raw = self.personalized_responses(features, requests)
        logits = self.gate_logits(features, requests)
        if forced_mask is not None:
            mask = forced_mask.to(dtype=raw.dtype)
        elif force_all_links:
            mask = torch.ones_like(logits[..., :1])
        else:
            mask = (torch.softmax(logits, dim=-1)[..., :1] > 0.5).to(raw.dtype)
        eye = torch.eye(self.n_agents, device=raw.device, dtype=raw.dtype).view(
            1, self.n_agents, self.n_agents, 1,
        )
        mask = mask * (1.0 - eye)
        received = (raw * mask).transpose(1, 2)
        received = received.masked_select(
            ~(eye.transpose(1, 2).bool().expand_as(received)),
        ).view(raw.shape[0], self.n_agents, self.n_agents - 1, self.response_dim)
        # The released implementation removes the unused diagonal before LSQ;
        # including it would change LSQ's gradient scale despite carrying no data.
        received = self.response_quantizer(received)
        return received, logits, mask

    def policy(self, features: Tensor, received: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Paper Equation (3): aggregate local features/messages then update Q history."""
        feature_key, feature_query, feature_value = self.policy_feature_kqv(features).chunk(3, dim=-1)
        message_key, message_query, message_value = self.policy_message_kqv(received).chunk(3, dim=-1)
        key = torch.cat([feature_key, message_key], dim=2)
        query = torch.cat([feature_query, message_query], dim=2)
        value = torch.cat([feature_value, message_value], dim=2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.encode_dim**0.5
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        attended[:, :, : self.n_entities] += features
        policy_input = self.policy_input(attended.flatten(2))
        next_hidden = self.policy_gru(
            policy_input.flatten(0, 1), hidden.flatten(0, 1),
        ).view_as(hidden)
        return self.q_head(next_hidden), next_hidden

    def step(
        self,
        obs: Tensor,
        hidden: Tensor,
        *,
        force_all_links: bool = False,
        forced_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        features, requests = self.encode(obs, hidden)
        received, gate_logits, gate_mask = self.communication(
            features, requests, force_all_links=force_all_links, forced_mask=forced_mask,
        )
        q_values, next_hidden = self.policy(features, received, hidden)
        return q_values, next_hidden, {
            "features": features, "requests": requests, "responses": received,
            "gate_logits": gate_logits, "gate_mask": gate_mask,
        }

    def helper_value_loss(self, products: dict[str, Tensor], q_values: Tensor) -> Tensor:
        """Paper Equation (14): predict each helper's detached local Q at receivers."""
        features, received = products["features"], products["responses"]
        feature_key, feature_value = self.predict_feature_kv(features).chunk(2, dim=-1)
        message_key, query, message_value = self.predict_message_kqv(received).chunk(3, dim=-1)
        key = torch.cat([feature_key, message_key], dim=2)
        value = torch.cat([feature_value, message_value], dim=2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.encode_dim**0.5
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        predictions = self.predict_q(attended.flatten(2)).view(
            q_values.shape[0], self.n_agents, self.n_agents - 1, self.action_dim,
        )
        targets = q_values.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        eye = torch.eye(self.n_agents, device=q_values.device, dtype=torch.bool)
        targets = targets.masked_select(
            ~eye.view(1, self.n_agents, self.n_agents, 1).expand_as(targets),
        ).view_as(predictions)
        return F.mse_loss(predictions, targets.detach(), reduction="none").mean(
            dim=(1, 2, 3),
        )

    def gate_labels(
        self, obs: Tensor, hidden: Tensor, helper: int, threshold: float = 0.0,
        *, mode: str = "release", mixer: nn.Module | None = None, state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Class 0 communicates; class 1 prunes the link.

        ``release`` compares maxima of two different local value functions.
        ``paper`` compares the two actions under the same link-on value function
        and mixer, holding other agents' actions fixed (paper Equations 7--11).
        It requires the centralized state and mixer used by the learner. The
        common link-on context avoids comparing values with different message
        inputs instead of measuring the value of the changed action.
        """
        if mode not in {"release", "paper"}:
            raise ValueError("gate mode must be release or paper")
        if mode == "paper" and (mixer is None or state is None):
            raise ValueError("paper gate labels require the centralized state and mixer")
        with torch.no_grad():
            features, requests = self.encode(obs, hidden)
            _, _, base_mask = self.communication(features, requests)
        logits = self.gate_logits(features.detach(), requests.detach())
        receivers = [index for index in range(self.n_agents) if index != helper]
        on_mask, off_mask = base_mask.clone(), base_mask.clone()
        on_mask[:, helper, receivers] = 1.0
        off_mask[:, helper, receivers] = 0.0
        with torch.no_grad():
            on_received, _, _ = self.communication(features, requests, forced_mask=on_mask)
            off_received, _, _ = self.communication(features, requests, forced_mask=off_mask)
            q_on, _ = self.policy(features, on_received, hidden)
            q_off, _ = self.policy(features, off_received, hidden)
            on_values = q_on[:, receivers]
            off_values = q_off[:, receivers]
            if mode == "paper":
                alternatives = off_values.argmax(dim=-1, keepdim=True)
                off_value = on_values.gather(-1, alternatives).squeeze(-1)
                chosen = q_on.max(dim=-1).values
                full_value = mixer(chosen, state).reshape(-1)
                improvements = []
                for index, receiver in enumerate(receivers):
                    counterfactual = chosen.clone()
                    counterfactual[:, receiver] = off_value[:, index]
                    improvements.append(full_value - mixer(counterfactual, state).reshape(-1))
                improvement = torch.stack(improvements, dim=-1)
            else:
                off_value = off_values.max(dim=-1).values
                improvement = on_values.max(dim=-1).values - off_value
            labels = (improvement <= threshold).long()
        return logits[:, helper, receivers], labels


class CACOMAgent(nn.Module):
    """CACOM online/target networks and the paper's QMIX backbone."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        *,
        input_dim: int | None = None,
        entity_schema: tuple[tuple[int, int], ...] | None = None,
        encode_dim: int = 8,
        request_dim: int = 4,
        response_dim: int = 8,
        hidden_dim: int = 64,
        bits: int = 2,
        mixer_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network = CACOMNetwork(
            n_agents, input_dim or obs_dim, action_dim, entity_schema=entity_schema,
            encode_dim=encode_dim, request_dim=request_dim,
            response_dim=response_dim, hidden_dim=hidden_dim, bits=bits,
        )
        self.mixer = QMixer(
            n_agents, n_agents * obs_dim, mixer_hidden_dim,
            hypernet_hidden_dim=64,
        )
        self.target_network = copy.deepcopy(self.network)
        self.target_mixer = copy.deepcopy(self.mixer)

    def init_hidden(self, batch: int, device: torch.device) -> Tensor:
        return self.network.init_hidden(batch, device)

    def gate_parameters(self) -> list[nn.Parameter]:
        return list(self.network.gate_key.parameters()) + list(
            self.network.gate_query.parameters()
        ) + list(self.network.gate_head.parameters())

    def policy_parameters(self) -> list[nn.Parameter]:
        gate_ids = {id(parameter) for parameter in self.gate_parameters()}
        return [
            parameter for parameter in self.network.parameters()
            if id(parameter) not in gate_ids
        ] + list(self.mixer.parameters())

    def update_targets(self) -> None:
        """Sync targets, leaving the target gate at initialization.

        The release owns ``ExpGate`` on the controller, outside the agent, and
        ``load_state`` copies the agent only — so the target gate never advances past
        its initialization even though target Q-values are computed with it.
        """
        gate_ids = {id(parameter) for parameter in self.gate_parameters()}
        gate_names = {
            name for name, parameter in self.network.named_parameters()
            if id(parameter) in gate_ids
        }
        state = {
            name: tensor for name, tensor in self.network.state_dict().items()
            if name not in gate_names
        }
        self.target_network.load_state_dict(state, strict=False)
        self.target_mixer.load_state_dict(self.mixer.state_dict())


__all__ = ["CACOMAgent", "CACOMNetwork", "LearnedStepQuantizer"]
