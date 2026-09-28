"""ExpoComm implementation.

Original paper:
Xinran Li, Xiaolu Wang, Chenjia Bai, and Jun Zhang. "Exponential
Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning."
International Conference on Learning Representations (ICLR), 2025.
Official code: https://github.com/LXXXXR/ExpoComm (Apache-2.0), reference commit
25dc9729c0ac65a283ae76977cfbec6df41249c0.

Model: a shared recurrent Q-network maintains one local hidden state and one
message memory per agent. The static variant adds its local hidden state to an
attention aggregation of previous messages at self and every power-of-two offset.
The one-peer variant preserves each agent's current local information and rotates
through one non-self power-of-two peer, folding that peer's previous message into
memory with a GRU. A QMIX mixer combines the per-agent values for the IMP configuration;
MAgent runs on IDQN, so `use_mixer=False` leaves them unmixed.
Invariants: topology is deterministic and near-linear in N; hidden/message order
is always agent-index order; targets are hard copies of all trainable modules.
Interface: ``step`` performs one recurrent communication/Q step and
``predict_state`` grounds messages by reconstructing the global state.

Paper/code reconciliation: Equation (2) writes a persistent self-loop plus one
rotating peer, while the released controller cycles through a self-only round and
then the power-of-two peers. The paper controls the communication semantics here,
so the self-only round is omitted and local information remains persistent through
the local recurrent stream. The released
static processor initializes message memory to zero and aggregates only that
memory, so its messages remain observation-independent; the local-hidden residual
here supplies the paper's stated accumulation of new information while preserving
the released exponential-neighbor attention. Contrastive
grounding samples at most 20 valid negatives with an explicit generator, matching
the released training objective while keeping seeded runs reproducible. Grounding follows the paper's own rule: Equation (4) needs a compact global state, and
Equation (5) is for environments whose state is "a concatenation of all observations and
is not compact or suitable for message grounding". modMARL's environments are the latter,
so `contrastive` is the default; `state` remains available and regresses onto the
concatenated joint observation. Two divergences remain in the contrastive estimator and
are deliberate rather than overlooked: the release draws one reference timestep for the
whole batch uniformly over all steps (padded included) and discards invalid rows
afterwards, whereas this samples per batch element from valid timesteps only; and the
temporal interval uses the paper's diameter `ceil(log2(N-1))` where the release uses
`topk_neighbors - 1`, which agree for every published agent count but differ at
N = 2^k + 1.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.nn import build_mlp
from .qmix import QMixer


def exponential_offsets(n_agents: int) -> tuple[int, ...]:
    """Return self plus power-of-two offsets below ``n_agents``."""
    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    return (0,) + tuple(1 << power for power in range(math.ceil(math.log2(n_agents))))


def exponential_peer_indices(n_agents: int, step: int, *, device: torch.device | None = None) -> Tensor:
    """Sender index read by each receiver at ``step``; shape ``(n_agents,)``.

    Paper Equation (2) keeps self-information in the local stream and rotates through
    the non-self power-of-two offsets. The released controller's extra self-only round
    is intentionally omitted.
    """
    offsets = exponential_offsets(n_agents)
    peers = offsets[1:]
    offset = peers[step % len(peers)] if peers else 0
    return (torch.arange(n_agents, device=device) + offset) % n_agents


def static_exponential_peer_indices(
    n_agents: int, *, device: torch.device | None = None,
) -> Tensor:
    """All static exponential senders for each receiver; shape ``(N, K)``."""
    offsets = torch.as_tensor(exponential_offsets(n_agents), device=device)
    receivers = torch.arange(n_agents, device=device).unsqueeze(-1)
    return (receivers + offsets) % n_agents


class ExpoCommNetwork(nn.Module):
    """Shared recurrent Q-network with static or one-peer ExpoComm messaging."""

    def __init__(
        self, input_dim: int, action_dim: int, state_dim: int, hidden_dim: int = 64,
        topology: str = "one_peer", attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        if topology not in {"one_peer", "static"}:
            raise ValueError("topology must be 'one_peer' or 'static'")
        self.hidden_dim = hidden_dim
        self.topology = topology
        self.attention_dim = attention_dim or hidden_dim
        self.obs_encoder = nn.Linear(input_dim, hidden_dim)
        self.obs_gru = nn.GRUCell(hidden_dim, hidden_dim)
        if topology == "one_peer":
            self.message_input = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
            self.message_gru = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.message_query = nn.Linear(hidden_dim, self.attention_dim)
            self.message_key = nn.Linear(hidden_dim, self.attention_dim)
            self.message_value = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = build_mlp(2 * hidden_dim, [hidden_dim], action_dim)
        self.state_predictor = build_mlp(hidden_dim, [hidden_dim], state_dim)

    def step(
        self,
        obs: Tensor,
        hidden: Tensor,
        messages: Tensor,
        peer_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance one timestep.

        ``obs`` is ``(B, N, obs_dim)`` and recurrent inputs are ``(B, N, H)``.
        ``peer_indices`` is ``(N,)`` or ``(B, N)`` and maps receivers to senders.
        Returns Q-values, next local hidden states, and next message memories.
        """
        batch, n_agents, _ = obs.shape
        encoded = torch.relu(self.obs_encoder(obs)).reshape(batch * n_agents, self.hidden_dim)
        next_hidden = self.obs_gru(encoded, hidden.reshape(batch * n_agents, self.hidden_dim))
        next_hidden = next_hidden.view(batch, n_agents, self.hidden_dim)

        if self.topology == "one_peer":
            next_messages = self._one_peer_messages(next_hidden, messages, peer_indices)
        else:
            next_messages = self._static_messages(next_hidden, messages, peer_indices)
        q_values = self.q_head(torch.cat([next_hidden, next_messages], dim=-1))
        return q_values, next_hidden, next_messages

    def _one_peer_messages(
        self, hidden: Tensor, messages: Tensor, peer_indices: Tensor,
    ) -> Tensor:
        batch, n_agents, _ = messages.shape
        if peer_indices.ndim == 1:
            peer_indices = peer_indices.unsqueeze(0).expand(batch, -1)
        gather = peer_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        received = messages.gather(1, gather)
        message_input = self.message_input(torch.cat([hidden, received], dim=-1))
        return self.message_gru(
            message_input.reshape(batch * n_agents, self.hidden_dim),
            messages.reshape(batch * n_agents, self.hidden_dim),
        ).view(batch, n_agents, self.hidden_dim)

    def _static_messages(
        self, hidden: Tensor, messages: Tensor, peer_indices: Tensor,
    ) -> Tensor:
        batch, n_agents, _ = messages.shape
        if peer_indices.ndim == 2:
            peer_indices = peer_indices.unsqueeze(0).expand(batch, -1, -1)
        n_peers = peer_indices.shape[-1]
        source = messages.unsqueeze(1).expand(-1, n_agents, -1, -1)
        gather = peer_indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        received = source.gather(2, gather)
        query = self.message_query(hidden).unsqueeze(-1)
        keys = self.message_key(received)
        scores = torch.matmul(keys, query).squeeze(-1) / self.attention_dim**0.5
        weights = torch.softmax(scores, dim=-1)
        values = self.message_value(received)
        assert weights.shape == (batch, n_agents, n_peers)
        return hidden + (weights.unsqueeze(-1) * values).sum(dim=2)

    def predict_state(self, messages: Tensor) -> Tensor:
        """Predict the flattened global state independently from each message."""
        return self.state_predictor(messages)


class ExpoCommAgent(nn.Module):
    """ExpoComm online/target recurrent networks and optional QMIX mixers."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        mixer_hidden_dim: int = 32,
        attention_dim: int = 16,
        topology: str = "one_peer",
        grounding: str = "state",
        use_mixer: bool = True,
    ) -> None:
        super().__init__()
        if grounding not in {"state", "contrastive"}:
            raise ValueError("grounding must be 'state' or 'contrastive'")
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.topology = topology
        self.grounding = grounding
        state_dim = n_agents * obs_dim
        input_dim = obs_dim + action_dim + n_agents
        self.network = ExpoCommNetwork(
            input_dim, action_dim, state_dim, hidden_dim, topology=topology,
            attention_dim=attention_dim,
        )
        # MAgent runs on IDQN in the paper and sets `mixer:` empty in every released
        # config; IMP runs on QMIX. `use_mixer=False` selects the former.
        self.mixer = QMixer(
            n_agents, state_dim, mixer_hidden_dim, hypernet_hidden_dim=64,
        ) if use_mixer else None
        self.target_network = copy.deepcopy(self.network)
        self.target_mixer = copy.deepcopy(self.mixer)

    def mix(self, values: Tensor, state: Tensor, *, target: bool = False) -> Tensor:
        """Mix per-agent values, or return them unmixed under the IDQN configuration."""
        mixer = self.target_mixer if target else self.mixer
        if mixer is None:
            return values
        batch, steps, n_agents = values.shape
        return mixer(
            values.reshape(-1, n_agents), state.reshape(batch * steps, -1),
        ).view(batch, steps)

    def init_recurrent(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
        zeros = torch.zeros(batch, self.n_agents, self.hidden_dim, device=device)
        return zeros, zeros.clone()

    def step(
        self,
        obs: Tensor,
        hidden: Tensor,
        messages: Tensor,
        timestep: int,
        *,
        previous_actions: Tensor | None = None,
        target: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        network = self.target_network if target else self.network
        if self.topology == "one_peer":
            peers = exponential_peer_indices(self.n_agents, timestep, device=obs.device)
        else:
            peers = static_exponential_peer_indices(self.n_agents, device=obs.device)
        if previous_actions is None:
            previous_actions = obs.new_zeros(obs.shape[0], self.n_agents, self.action_dim)
        agent_ids = torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype)
        agent_ids = agent_ids.unsqueeze(0).expand(obs.shape[0], -1, -1)
        inputs = torch.cat([obs, previous_actions, agent_ids], dim=-1)
        return network.step(inputs, hidden, messages, peers)

    def predict_state(self, messages: Tensor) -> Tensor:
        return self.network.predict_state(messages)

    def grounding_loss(
        self,
        messages: Tensor,
        states: Tensor | None = None,
        mask: Tensor | None = None,
        *,
        temperature: float = 0.07,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Paper Equations (4)--(5) for ``messages: (B, T, N, H)``.

        State grounding predicts the current global state from every agent message.
        Contrastive grounding uses other agents at the same timestep as positives
        and timesteps outside the exponential graph diameter as negatives.
        """
        if self.grounding == "state":
            if states is None:
                raise ValueError("states are required for state grounding")
            predictions = self.predict_state(messages)
            targets = states.unsqueeze(2).expand_as(predictions)
            error = (predictions - targets).pow(2)
            if mask is None:
                return error.mean()
            # The release sums the squared error over state_dim but normalizes by a
            # mask that spans only (batch, time, agent), so the denominator counts
            # predictions, not scalars. Expanding it over state_dim as well would shrink
            # the loss by exactly state_dim.
            weights = mask.unsqueeze(-1).unsqueeze(-1)
            predictions_counted = weights.sum().clamp_min(1.0) * error.shape[2]
            return (error * weights).sum() / predictions_counted
        return expo_contrastive_loss(
            messages, mask=mask, diameter=math.ceil(math.log2(max(1, self.n_agents - 1))),
            temperature=temperature, generator=generator,
        )

    def update_targets(self) -> None:
        """Hard-copy the online network and mixer into their target copies."""
        self.target_network.load_state_dict(self.network.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())


__all__ = [
    "ExpoCommAgent",
    "ExpoCommNetwork",
    "expo_contrastive_loss",
    "exponential_offsets",
    "exponential_peer_indices",
    "static_exponential_peer_indices",
]


def expo_contrastive_loss(
    messages: Tensor,
    *,
    mask: Tensor | None,
    diameter: int,
    temperature: float = 0.07,
    max_negatives: int = 20,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Released sampled-negative form of ExpoComm's InfoNCE objective.

    Every other agent at the same valid timestep is a positive. Valid messages
    more than ``diameter`` timesteps away are negatives and are sampled down to
    ``max_negatives`` using ``generator``.
    """
    if messages.ndim != 4:
        raise ValueError("messages must have shape (batch, time, agents, features)")
    batch, steps, n_agents, _ = messages.shape
    if n_agents < 2:
        raise ValueError("contrastive grounding requires at least two agents")
    if steps <= 2 * diameter + 1:
        raise ValueError("sequence is too short to contain temporal negatives")
    normalized = F.normalize(messages, dim=-1)
    valid = torch.ones(batch, steps, dtype=torch.bool, device=messages.device)
    if mask is not None:
        valid = mask.to(dtype=torch.bool)
    losses: list[Tensor] = []
    for batch_index in range(batch):
        valid_times = valid[batch_index].nonzero(as_tuple=False).flatten()
        reference_candidates = [
            int(time) for time in valid_times
            if any(abs(int(other) - int(time)) > diameter for other in valid_times)
        ]
        if not reference_candidates:
            continue
        reference_pick = torch.randint(
            len(reference_candidates), (), device=messages.device, generator=generator,
        )
        reference_time = reference_candidates[int(reference_pick)]
        negative_times = [
            int(time) for time in valid_times
            if abs(int(time) - reference_time) > diameter
        ]
        negative_pick = torch.randint(
            len(negative_times), (), device=messages.device, generator=generator,
        )
        negative_time = negative_times[int(negative_pick)]

        anchors = normalized[batch_index, reference_time]
        positive_offsets = torch.randint(
            1, n_agents, (n_agents,), device=messages.device, generator=generator,
        )
        positive_ids = (
            torch.arange(n_agents, device=messages.device) + positive_offsets
        ) % n_agents
        positives = normalized[batch_index, reference_time, positive_ids]
        negative_ids = torch.randperm(
            n_agents, device=messages.device, generator=generator,
        )[:max_negatives]
        negatives = normalized[batch_index, negative_time, negative_ids]
        positive_logits = (anchors * positives).sum(-1, keepdim=True)
        negative_logits = positives @ negatives.transpose(0, 1)
        logits = torch.cat([positive_logits, negative_logits], dim=-1) / temperature
        losses.extend((-F.log_softmax(logits, dim=-1)[:, 0]).unbind())
    if not losses:
        raise ValueError("no valid positive/negative contrastive pairs")
    return torch.stack(losses).mean()
