"""Interactive World Latent (IWoL) representation learning and MAPPO.

Paper: Dongsu Lee, Daehee Lee, Yaru Niu, Honguk Woo, Amy Zhang, and Ding
Zhao, "Learning to Interact in World Latent for Team Coordination,"
arXiv:2509.25550v3 (v4 was retitled "Unifying Agent Interaction and World
Information for Multi-agent Coordination").  Pinned source: DongsuLeeTech/IWoL at
de3bc5b1e50bd9c4d90672a6355269ea2917fd28.

Model: shared recurrent actors encode local observations; Im-IWoL learns a
message-free latent supervised by privileged-state and critic-message decoders,
while Ex-IWoL sends dynamically scheduled Transformer messages.  Both variants
use a decentralized communication critic and recurrent MAPPO.  Tensor contracts
retain joint agent axes: observations are (batch, agents, observation), graphs
are (batch, receiver, sender), and rollout chunks are (chunk, time, agents, ...).

Invariants: graph diagonals are self links, the learned graph is intersected with
the physical proximity graph, masked senders have exactly zero attention weight,
PPO replays behavior-time Gumbel noise, and Im-IWoL detaches the critic message
teacher.  Interface: IWoLAgent owns collection and optimization; IWoLRollout
stores complete recurrent episodes and never shuffles communicating agents.

Source reconciliation: the paper controls the graph equations, the release the network and
training details.  On the value loss the release and paper Table E.5 agree (Huber, delta
10, clip 10) against Appendix C.2's MSE and clip 0.5, so Table E.5 governs.

Three released defects are corrected, each pinned by a golden test: `transformer_comm.py`
multiplies pre-softmax logits by the 0/1 graph, so masked senders keep weight; the
`comm_graphs` physical graph argument is threaded everywhere and never read, though paper
Section 4 requires G = G_c * G_p (only MetaDrive supplies one); and `evaluate_actions` and
`latent_world` each resample the scheduler, so behavior-time Gumbel noise is replayed here
instead.

Paper/code reconciliation: the world decoder regresses the joint observation as a privileged
state per paper Section 4.3, where `use_centralized_V: False` makes the release
reconstruct each agent's own observation; the Im-IWoL policy consumes `(features, latent)`
as the release does rather than the latent alone of Algorithm 1; world and
value-normalizer statistics are masked by validity where upstream is unmasked.

Validation uses dependency-free cooperative navigation as bounded learning evidence, not a
reproduction of the paper's robotics benchmark tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mappo import ValueNorm, huber_loss


def _orthogonal(linear: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def _position_dim(position_slice: tuple[int, int] | None) -> int | None:
    """Width of the released positional-embedding slice, or None when disabled."""
    if position_slice is None:
        return None
    start, end = position_slice
    if not 0 <= start < end:
        raise ValueError("position_slice must satisfy 0 <= start < end")
    return end - start


def _positions(obs: Tensor, position_slice: tuple[int, int] | None) -> Tensor | None:
    """Released `obs[start:end]` positional features.

    The upstream config indices are one-based (`obs_pos_embed_start 20` slices
    `obs[19:21]`); `position_slice` here is an ordinary zero-based Python slice.
    """
    if position_slice is None:
        return None
    return obs[..., position_slice[0]:position_slice[1]]


def _sample_gumbel(shape: torch.Size, reference: Tensor, generator: torch.Generator | None) -> Tensor:
    uniform = torch.rand(shape, dtype=reference.dtype, device=reference.device, generator=generator)
    return -torch.log(-torch.log(uniform.clamp_(1e-6, 1.0 - 1e-6)))


class IWoLScheduler(nn.Module):
    """Paper additive-attention scheduler with hard reparameterized links."""

    def __init__(
        self,
        feature_dim: int,
        n_agents: int,
        *,
        n_heads: int = 1,
        negative_slope: float = 1.2,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if n_agents < 1 or n_heads < 1 or temperature <= 0:
            raise ValueError("n_agents, n_heads, and temperature must be positive")
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.negative_slope = negative_slope
        self.temperature = temperature
        # Published runs feed `rnn_enc` through the release's feature-normalized
        # MLPBase with layer_N=1 before additive attention.
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            _orthogonal(nn.Linear(feature_dim, feature_dim), nn.init.calculate_gain("relu")),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
            _orthogonal(nn.Linear(feature_dim, feature_dim), nn.init.calculate_gain("relu")),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        )
        self.receiver_weight = nn.Parameter(torch.empty(n_heads, feature_dim))
        self.sender_weight = nn.Parameter(torch.empty(n_heads, feature_dim))
        nn.init.xavier_normal_(self.receiver_weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.sender_weight, gain=nn.init.calculate_gain("relu"))

    def link_logits(self, features: Tensor) -> Tensor:
        """Return communicate/no-communicate logits as ``(B,N,N,2)``."""
        if features.ndim != 3 or features.shape[1] != self.n_agents:
            raise ValueError(
                f"features must have shape (batch, {self.n_agents}, feature_dim)"
            )
        encoded = self.feature_encoder(features)
        receiver = torch.einsum("bnd,hd->bhn", encoded, self.receiver_weight)
        sender = torch.einsum("bnd,hd->bhn", encoded, self.sender_weight)
        scores = (receiver.unsqueeze(-1) + sender.unsqueeze(-2)).mean(dim=1)
        link = F.leaky_relu(scores, negative_slope=self.negative_slope)
        return torch.stack((link, torch.zeros_like(link)), dim=-1)

    def forward(
        self,
        features: Tensor,
        *,
        noise: Tensor | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return a hard graph ``(B,N,N)`` and replayable noise ``(B,N,N,2)``."""
        logits = self.link_logits(features)
        if deterministic:
            graph = logits.argmax(dim=-1).eq(0).to(features.dtype)
            return graph, torch.zeros_like(logits)
        if noise is None:
            noise = _sample_gumbel(logits.shape, logits, generator)
        elif noise.shape != logits.shape:
            raise ValueError(f"noise must have shape {tuple(logits.shape)}")
        probabilities = ((logits + noise) / self.temperature).softmax(dim=-1)
        hard = F.one_hot(probabilities.argmax(dim=-1), num_classes=2).to(probabilities.dtype)
        straight_through = hard + probabilities - probabilities.detach()
        return straight_through[..., 0], noise


def compose_communication_graph(learned: Tensor, physical: Tensor | None) -> Tensor:
    """Apply physical feasibility and force self-information in the final graph."""
    if learned.ndim != 3 or learned.shape[-1] != learned.shape[-2]:
        raise ValueError("learned graph must have shape (batch, agents, agents)")
    if physical is None:
        physical = torch.ones_like(learned)
    if physical.shape != learned.shape:
        raise ValueError("physical graph must match the learned graph")
    physical = physical.to(dtype=learned.dtype, device=learned.device)
    agents = learned.shape[-1]
    identity = torch.eye(agents, dtype=learned.dtype, device=learned.device).unsqueeze(0)
    off_diagonal = 1.0 - identity
    return learned * physical * off_diagonal + identity


class MaskedMessageAttention(nn.Module):
    """Multi-head attention whose graph mask has exact zeros and useful gradients."""

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.key = _orthogonal(nn.Linear(hidden_dim, hidden_dim), 0.01)
        self.query = _orthogonal(nn.Linear(hidden_dim, hidden_dim), 0.01)
        self.value = _orthogonal(nn.Linear(hidden_dim, hidden_dim), 0.01)
        self.output = _orthogonal(nn.Linear(hidden_dim, hidden_dim), 0.01)

    def forward(self, hidden: Tensor, graph: Tensor) -> tuple[Tensor, Tensor]:
        """Aggregate sender values for receiver rows; shapes are ``(B,N,H)`` and ``(B,N,N)``."""
        batch, agents, width = hidden.shape
        head_width = width // self.n_heads
        q = self.query(hidden).view(batch, agents, self.n_heads, head_width).transpose(1, 2)
        k = self.key(hidden).view(batch, agents, self.n_heads, head_width).transpose(1, 2)
        v = self.value(hidden).view(batch, agents, self.n_heads, head_width).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)

        # Multiplication followed by renormalization equals a true masked softmax
        # in the forward pass and preserves the straight-through graph derivative.
        weights = scores.softmax(dim=-1) * graph.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, agents, width)
        return self.output(attended), weights


class _CommunicationBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attention = MaskedMessageAttention(hidden_dim, n_heads)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            _orthogonal(nn.Linear(hidden_dim, hidden_dim), nn.init.calculate_gain("relu")),
            nn.GELU(),
            _orthogonal(nn.Linear(hidden_dim, hidden_dim), 0.01),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, graph: Tensor) -> tuple[Tensor, Tensor]:
        message, weights = self.attention(hidden, graph)
        hidden = self.attention_norm(hidden + message)
        return self.output_norm(hidden + self.feed_forward(hidden)), weights


class IWoLCommunication(nn.Module):
    """Released Transformer message processor with paper-correct graph masking."""

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        *,
        n_heads: int = 4,
        n_hops: int = 4,
        position_dim: int | None = None,
    ) -> None:
        super().__init__()
        if n_hops < 1:
            raise ValueError("n_hops must be positive")
        # `pos_embed: True` in every published config.
        self.pos_encoder = (
            nn.Linear(position_dim, hidden_dim) if position_dim else None
        )
        self.message_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            _orthogonal(nn.Linear(feature_dim, hidden_dim), nn.init.calculate_gain("relu")),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [_CommunicationBlock(hidden_dim, n_heads) for _ in range(n_hops)]
        )
        self.message_head = nn.Sequential(
            _orthogonal(nn.Linear(hidden_dim, hidden_dim), nn.init.calculate_gain("relu")),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            _orthogonal(nn.Linear(hidden_dim, feature_dim), 0.01),
        )

    def forward(
        self, features: Tensor, graph: Tensor, positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.message_encoder(features)
        if self.pos_encoder is not None and positions is not None:
            hidden = hidden + self.pos_encoder(positions)
        hidden = self.input_norm(hidden)
        weights = graph.unsqueeze(1)
        for block in self.blocks:
            hidden, weights = block(hidden, graph)
        return self.message_head(hidden), weights


def communication_rate(graph: Tensor) -> Tensor:
    """Return non-self directed link utilization averaged over leading batches."""
    if graph.ndim < 2:
        raise ValueError("graph must end in square agent axes")
    agents = graph.shape[-1]
    if graph.shape[-2:] != (agents, agents):
        raise ValueError("graph must end in square agent axes")
    if agents == 1:
        return graph.new_zeros(())
    identity = torch.eye(agents, dtype=graph.dtype, device=graph.device)
    possible_links = graph.numel() / (agents * agents) * agents * (agents - 1)
    return (graph * (1.0 - identity)).sum() / possible_links


class IWoLMode(str, Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class IWoLActionKind(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class LocalObservationEncoder(nn.Module):
    """Released ``MLPBase`` feature encoder followed by the normalized recurrent layer.

    Reproduces ``MLPBase`` -- ``LayerNorm(obs)`` then ``1 + layer_N`` blocks of
    ``Linear -> ReLU -> LayerNorm`` -- and ``RNNLayer``, whose output ``LayerNorm`` the
    features pass through while the raw cell state is carried forward.

    The paper says the observation is "processed by a self-attention layer", but no
    released variant implements one: ``obs_enc_type`` offers an ``'attention'`` choice in
    the parser and is never branched on, every published config selects ``rnn``, and the
    paper's Appendix C.2 states the component "can be replaced on the recurrent neural
    network variants". The release governs here.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_encoder_layers: int = 1,
    ) -> None:
        super().__init__()
        if n_encoder_layers < 0:
            raise ValueError("n_encoder_layers must be non-negative")
        self.obs_dim = obs_dim
        gain = nn.init.calculate_gain("relu")
        layers: list[nn.Module] = [
            nn.LayerNorm(obs_dim),
            _orthogonal(nn.Linear(obs_dim, hidden_dim), gain),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(n_encoder_layers):
            layers += [
                _orthogonal(nn.Linear(hidden_dim, hidden_dim), gain),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ]
        self.feature = nn.Sequential(*layers)
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
        for name, parameter in self.recurrent.named_parameters():
            nn.init.zeros_(parameter) if "bias" in name else nn.init.orthogonal_(parameter)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs: Tensor, hidden: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``obs: (B,N,O)`` and advance ``hidden: (B,N,H)``."""
        if obs.ndim != 3 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs must have shape (batch, agents, {self.obs_dim})")
        if hidden.shape[:2] != obs.shape[:2] or masks.shape != obs.shape[:2]:
            raise ValueError("hidden and masks must match the observation batch and agent axes")
        encoded = self.feature(obs)
        batch, agents, width = encoded.shape
        next_hidden = self.recurrent(
            encoded.reshape(batch * agents, width),
            (hidden * masks.unsqueeze(-1)).reshape(batch * agents, width),
        ).view(batch, agents, width)
        return self.output_norm(next_hidden), next_hidden


class InteractionWorldEncoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _orthogonal(nn.Linear(hidden_dim, hidden_dim // 2), nn.init.calculate_gain("relu")),
            nn.ReLU(),
            _orthogonal(nn.Linear(hidden_dim // 2, latent_dim), 1.0),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class IWoLDecoder(nn.Module):
    """Released three-layer bounded decoder used by both auxiliary objectives."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.first = _orthogonal(nn.Linear(input_dim, hidden_dim), 1.0)
        self.second = _orthogonal(nn.Linear(hidden_dim, hidden_dim), 1.0)
        self.output = _orthogonal(nn.Linear(hidden_dim, output_dim), 0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.tanh(self.output(torch.relu(self.second(torch.relu(self.first(inputs))))))


class IWoLActionHead(nn.Module):
    """Released categorical or state-independent diagonal-Gaussian policy head."""

    def __init__(self, input_dim: int, action_dim: int, kind: IWoLActionKind) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.kind = IWoLActionKind(kind)
        self.output = _orthogonal(nn.Linear(input_dim, action_dim), 0.01)
        if self.kind is IWoLActionKind.CONTINUOUS:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.register_parameter("log_std", None)

    def distribution(
        self, features: Tensor, available_actions: Tensor | None = None,
    ) -> torch.distributions.Categorical | torch.distributions.Normal:
        outputs = self.output(features)
        if self.kind is IWoLActionKind.DISCRETE:
            if available_actions is not None:
                outputs = outputs.masked_fill(
                    ~available_actions.bool(), torch.finfo(outputs.dtype).min,
                )
            return torch.distributions.Categorical(logits=outputs)
        if available_actions is not None:
            raise ValueError("available_actions apply only to discrete IWoL policies")
        assert self.log_std is not None
        return torch.distributions.Normal(outputs, self.log_std.exp())

    def sample(
        self,
        distribution: torch.distributions.Categorical | torch.distributions.Normal,
        deterministic: bool,
    ) -> Tensor:
        if isinstance(distribution, torch.distributions.Categorical):
            return distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        return distribution.mean if deterministic else distribution.sample()

    @staticmethod
    def log_prob(
        distribution: torch.distributions.Categorical | torch.distributions.Normal,
        actions: Tensor,
    ) -> Tensor:
        result = distribution.log_prob(actions)
        return result if isinstance(distribution, torch.distributions.Categorical) else result.sum(-1)

    @staticmethod
    def entropy(
        distribution: torch.distributions.Categorical | torch.distributions.Normal,
    ) -> Tensor:
        result = distribution.entropy()
        return result if isinstance(distribution, torch.distributions.Categorical) else result.sum(-1)


@dataclass(frozen=True)
class IWoLActorOutput:
    distribution: torch.distributions.Categorical | torch.distributions.Normal
    hidden: Tensor
    graph: Tensor
    noise: Tensor
    message: Tensor
    world_prediction: Tensor
    interaction_prediction: Tensor | None


@dataclass(frozen=True)
class IWoLCriticOutput:
    values: Tensor
    hidden: Tensor
    graph: Tensor
    noise: Tensor
    message: Tensor


class IWoLActor(nn.Module):
    """Shared local policy for one explicit IWoL execution mode."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        *,
        mode: IWoLMode,
        action_kind: IWoLActionKind,
        n_encoder_layers: int,
        hidden_dim: int,
        latent_dim: int,
        position_slice: tuple[int, int] | None,
        scheduler_heads: int,
        communication_heads: int,
        communication_hops: int,
        negative_slope: float,
        gumbel_temperature: float,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.mode = IWoLMode(mode)
        self.encoder = LocalObservationEncoder(
            obs_dim, hidden_dim, n_encoder_layers,
        )
        self.position_slice = position_slice
        if self.mode is IWoLMode.IMPLICIT:
            self.latent = InteractionWorldEncoder(hidden_dim, latent_dim)
            self.world_decoder = IWoLDecoder(latent_dim, hidden_dim, state_dim)
            self.interaction_decoder = IWoLDecoder(latent_dim, hidden_dim, hidden_dim)
            policy_dim = hidden_dim + latent_dim
        else:
            self.scheduler = IWoLScheduler(
                hidden_dim,
                n_agents,
                n_heads=scheduler_heads,
                negative_slope=negative_slope,
                temperature=gumbel_temperature,
            )
            self.communication = IWoLCommunication(
                hidden_dim, hidden_dim, n_heads=communication_heads, n_hops=communication_hops,
                position_dim=_position_dim(position_slice),
            )
            self.world_decoder = IWoLDecoder(hidden_dim, hidden_dim, state_dim)
            policy_dim = 2 * hidden_dim
        self.action_head = IWoLActionHead(policy_dim, action_dim, action_kind)

    def forward(
        self,
        obs: Tensor,
        hidden: Tensor,
        masks: Tensor,
        physical_graph: Tensor | None,
        available_actions: Tensor | None,
        *,
        noise: Tensor | None = None,
        deterministic_graph: bool = False,
        generator: torch.Generator | None = None,
    ) -> IWoLActorOutput:
        features, hidden = self.encoder(obs, hidden, masks)
        batch = obs.shape[0]
        if self.mode is IWoLMode.IMPLICIT:
            latent = self.latent(features)
            message = features.new_zeros(features.shape)
            graph = torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype).expand(
                batch, -1, -1,
            )
            replay_noise = features.new_zeros(batch, self.n_agents, self.n_agents, 2)
            policy_features = torch.cat((features, latent), dim=-1)
            world = self.world_decoder(latent)
            interaction = self.interaction_decoder(latent)
        else:
            learned, replay_noise = self.scheduler(
                features, noise=noise, deterministic=deterministic_graph, generator=generator,
            )
            graph = compose_communication_graph(learned, physical_graph)
            message, _ = self.communication(
                features, graph, _positions(obs, self.position_slice),
            )
            policy_features = torch.cat((features, message), dim=-1)
            world = self.world_decoder(message)
            interaction = None
        distribution = self.action_head.distribution(policy_features, available_actions)
        return IWoLActorOutput(
            distribution, hidden, graph, replay_noise, message, world, interaction,
        )


class IWoLCritic(nn.Module):
    """Communication-enabled decentralized value teacher used by both modes."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        *,
        n_encoder_layers: int,
        hidden_dim: int,
        position_slice: tuple[int, int] | None,
        scheduler_heads: int,
        communication_heads: int,
        communication_hops: int,
        negative_slope: float,
        gumbel_temperature: float,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.encoder = LocalObservationEncoder(
            obs_dim, hidden_dim, n_encoder_layers,
        )
        self.position_slice = position_slice
        self.scheduler = IWoLScheduler(
            hidden_dim,
            n_agents,
            n_heads=scheduler_heads,
            negative_slope=negative_slope,
            temperature=gumbel_temperature,
        )
        self.communication = IWoLCommunication(
            hidden_dim, hidden_dim, n_heads=communication_heads, n_hops=communication_hops,
            position_dim=_position_dim(position_slice),
        )
        self.value_head = _orthogonal(nn.Linear(2 * hidden_dim, 1), 1.0)

    def forward(
        self,
        obs: Tensor,
        hidden: Tensor,
        masks: Tensor,
        physical_graph: Tensor | None,
        *,
        noise: Tensor | None = None,
        deterministic_graph: bool = False,
        generator: torch.Generator | None = None,
    ) -> IWoLCriticOutput:
        features, hidden = self.encoder(obs, hidden, masks)
        learned, replay_noise = self.scheduler(
            features, noise=noise, deterministic=deterministic_graph, generator=generator,
        )
        graph = compose_communication_graph(learned, physical_graph)
        message, _ = self.communication(
            features, graph, _positions(obs, self.position_slice),
        )
        values = self.value_head(torch.cat((features, message), dim=-1)).squeeze(-1)
        return IWoLCriticOutput(values, hidden, graph, replay_noise, message)


@dataclass(frozen=True)
class IWoLStep:
    """One joint behavior step; every tensor retains its leading agent axis."""

    actions: Tensor
    log_probs: Tensor
    values: Tensor
    actor_hidden: Tensor
    critic_hidden: Tensor
    actor_graph: Tensor
    critic_graph: Tensor
    actor_noise: Tensor
    critic_noise: Tensor


@dataclass(frozen=True)
class IWoLBatch:
    """Padded recurrent chunks with leading dimensions ``(chunks,time,agents)``."""

    obs: Tensor
    states: Tensor
    physical_graphs: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    actor_hidden: Tensor
    critic_hidden: Tensor
    masks: Tensor
    active_masks: Tensor
    available_actions: Tensor
    actor_noise: Tensor
    critic_noise: Tensor
    valid: Tensor


class IWoLRollout:
    """Fresh episodes converted to joint-agent recurrent chunks."""

    _STEP_FIELDS = (
        "obs",
        "states",
        "physical_graphs",
        "actions",
        "old_log_probs",
        "old_values",
        "actor_hidden",
        "critic_hidden",
        "masks",
        "active_masks",
        "available_actions",
        "actor_noise",
        "critic_noise",
    )

    def __init__(self, learner: IWoLAgent) -> None:
        self.learner = learner
        self.episodes: list[IWoLBatch] = []
        self._current: dict[str, list[Tensor]] = {field: [] for field in self._STEP_FIELDS}
        self._rewards: list[Tensor] = []

    def add(
        self,
        *,
        obs: Tensor,
        states: Tensor,
        physical_graph: Tensor | None,
        step: IWoLStep,
        actor_hidden: Tensor,
        critic_hidden: Tensor,
        masks: Tensor,
        team_reward: float,
        active_masks: Tensor | None = None,
        available_actions: Tensor | None = None,
    ) -> None:
        """Append one transition collected from ``actor_hidden``/``critic_hidden``."""
        if physical_graph is None:
            physical_graph = torch.ones(
                self.learner.n_agents,
                self.learner.n_agents,
                dtype=obs.dtype,
                device=obs.device,
            )
        if active_masks is None:
            active_masks = torch.ones(self.learner.n_agents, dtype=obs.dtype, device=obs.device)
        if available_actions is None:
            available_actions = torch.ones(
                self.learner.n_agents,
                self.learner.action_dim,
                dtype=torch.bool,
                device=obs.device,
            )
        values = (
            obs,
            states,
            physical_graph,
            step.actions,
            step.log_probs,
            step.values,
            actor_hidden,
            critic_hidden,
            masks,
            active_masks,
            available_actions,
            step.actor_noise,
            step.critic_noise,
        )
        for field, value in zip(self._STEP_FIELDS, values):
            self._current[field].append(value.detach())
        self._rewards.append(torch.full_like(step.values, float(team_reward)))

    def finish_episode(self, bootstrap_value: Tensor, final_mask: Tensor) -> None:
        """Close the current episode; ``final_mask`` is zero only for true termination."""
        if not self._rewards:
            raise ValueError("cannot finish an empty IWoL episode")
        values = torch.stack(self._current["old_values"])
        rewards = torch.stack(self._rewards)
        transition_masks = torch.stack(self._current["masks"][1:] + [final_mask])
        advantages, returns = self.learner.compute_gae(
            rewards, values, bootstrap_value, transition_masks,
        )
        tensors = {field: torch.stack(items) for field, items in self._current.items()}
        tensors.update(
            advantages=advantages,
            returns=returns,
            valid=torch.ones_like(advantages),
        )
        self.episodes.append(IWoLBatch(**tensors))
        self._current = {field: [] for field in self._STEP_FIELDS}
        self._rewards.clear()

    def batch(self) -> IWoLBatch:
        """Return padded chunks without separating communicating agents."""
        if self._rewards or not self.episodes:
            raise RuntimeError("finish all IWoL episodes before requesting a batch")
        chunks: dict[str, list[Tensor]] = {field: [] for field in IWoLBatch.__dataclass_fields__}
        length = self.learner.chunk_length
        sequence_fields = tuple(
            field for field in IWoLBatch.__dataclass_fields__
            if field not in {"actor_hidden", "critic_hidden"}
        )
        for episode in self.episodes:
            horizon = episode.obs.shape[0]
            for start in range(0, horizon, length):
                stop = min(start + length, horizon)
                count = stop - start
                for field in sequence_fields:
                    value = getattr(episode, field)[start:stop]
                    padding = torch.zeros(
                        (length - count, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    chunks[field].append(torch.cat((value, padding)))
                chunks["actor_hidden"].append(episode.actor_hidden[start])
                chunks["critic_hidden"].append(episode.critic_hidden[start])
        return IWoLBatch(**{field: torch.stack(values) for field, values in chunks.items()})


@dataclass(frozen=True)
class _Unroll:
    log_probs: Tensor
    entropies: Tensor
    values: Tensor
    world_predictions: Tensor
    interaction_predictions: Tensor | None
    teacher_messages: Tensor


class IWoLAgent(nn.Module):
    """Complete shared-parameter IWoL learner for homogeneous teams."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        *,
        state_dim: int | None = None,
        mode: IWoLMode = IWoLMode.IMPLICIT,
        action_kind: IWoLActionKind = IWoLActionKind.DISCRETE,
        n_encoder_layers: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        position_slice: tuple[int, int] | None = None,
        scheduler_heads: int = 1,
        communication_heads: int = 4,
        communication_hops: int = 4,
        negative_slope: float = 1.2,
        gumbel_temperature: float = 0.1,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        world_coef: float = 0.05,
        interaction_coef: float = 0.05,
        max_grad_norm: float = 10.0,
        huber_delta: float = 10.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        chunk_length: int = 10,
    ) -> None:
        super().__init__()
        if n_agents < 1 or obs_dim < 1 or action_dim < 1 or chunk_length < 1:
            raise ValueError("agent, observation, action, and chunk dimensions must be positive")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim if state_dim is not None else n_agents * obs_dim
        self.mode = IWoLMode(mode)
        self.action_kind = IWoLActionKind(action_kind)
        shared = dict(
            n_encoder_layers=n_encoder_layers,
            hidden_dim=hidden_dim,
            position_slice=position_slice,
            scheduler_heads=scheduler_heads,
            communication_heads=communication_heads,
            communication_hops=communication_hops,
            negative_slope=negative_slope,
            gumbel_temperature=gumbel_temperature,
        )
        self.actor = IWoLActor(
            n_agents,
            obs_dim,
            self.state_dim,
            action_dim,
            mode=self.mode,
            action_kind=self.action_kind,
            latent_dim=latent_dim,
            **shared,
        )
        self.critic = IWoLCritic(n_agents, obs_dim, **shared)
        self.value_normalizer = ValueNorm()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-5, weight_decay=0.0,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5, weight_decay=0.0,
        )
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.world_coef = world_coef
        self.interaction_coef = interaction_coef
        self.max_grad_norm = max_grad_norm
        self.huber_delta = huber_delta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.chunk_length = chunk_length
        self.hidden_dim = hidden_dim

    def initial_state(
        self, device: torch.device, batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        shape = (self.n_agents, self.hidden_dim)
        if batch_size is not None:
            shape = (batch_size, *shape)
        zeros = torch.zeros(shape, device=device)
        return zeros, zeros.clone()

    @staticmethod
    def _batch_optional(value: Tensor | None, squeeze: bool) -> Tensor | None:
        return None if value is None else (value.unsqueeze(0) if squeeze else value)

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        actor_hidden: Tensor,
        critic_hidden: Tensor,
        masks: Tensor,
        physical_graph: Tensor | None = None,
        available_actions: Tensor | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> IWoLStep:
        """Sample joint actions and dynamic graphs from one recurrent step."""
        squeeze = obs.ndim == 2
        if squeeze:
            obs = obs.unsqueeze(0)
            actor_hidden = actor_hidden.unsqueeze(0)
            critic_hidden = critic_hidden.unsqueeze(0)
            masks = masks.unsqueeze(0)
        physical_graph = self._batch_optional(physical_graph, squeeze)
        available_actions = self._batch_optional(available_actions, squeeze)
        actor = self.actor(
            obs,
            actor_hidden,
            masks,
            physical_graph,
            available_actions if self.action_kind is IWoLActionKind.DISCRETE else None,
            deterministic_graph=deterministic,
            generator=generator,
        )
        critic = self.critic(
            obs,
            critic_hidden,
            masks,
            physical_graph,
            deterministic_graph=deterministic,
            generator=generator,
        )
        actions = self.actor.action_head.sample(actor.distribution, deterministic)
        log_probs = self.actor.action_head.log_prob(actor.distribution, actions)
        values = (
            actions,
            log_probs,
            critic.values,
            actor.hidden,
            critic.hidden,
            actor.graph,
            critic.graph,
            actor.noise,
            critic.noise,
        )
        if squeeze:
            values = tuple(value[0] for value in values)
        return IWoLStep(*values)

    @torch.no_grad()
    def values(
        self,
        obs: Tensor,
        critic_hidden: Tensor,
        masks: Tensor,
        physical_graph: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        squeeze = obs.ndim == 2
        if squeeze:
            obs, critic_hidden, masks = obs.unsqueeze(0), critic_hidden.unsqueeze(0), masks.unsqueeze(0)
        physical_graph = self._batch_optional(physical_graph, squeeze)
        # The release never takes argmax: `Scheduler.forward` has no `self.training`
        # branch, and `prep_rollout()`'s `policy.eval()` does not affect `gumbel_softmax`.
        output = self.critic(obs, critic_hidden, masks, physical_graph)
        return (output.values[0], output.hidden[0]) if squeeze else (output.values, output.hidden)

    def compute_gae(
        self,
        rewards: Tensor,
        normalized_values: Tensor,
        bootstrap_value: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values = self.value_normalizer.denormalize(normalized_values)
        next_value = self.value_normalizer.denormalize(bootstrap_value)
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(next_value)
        for step in range(rewards.shape[0] - 1, -1, -1):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
            next_value = values[step]
        return advantages, advantages + values

    def _unroll(self, batch: IWoLBatch, index: Tensor) -> _Unroll:
        actor_hidden = batch.actor_hidden[index]
        critic_hidden = batch.critic_hidden[index]
        log_probs, entropies, values = [], [], []
        worlds, interactions, messages = [], [], []
        for timestep in range(batch.obs.shape[1]):
            available = (
                batch.available_actions[index, timestep]
                if self.action_kind is IWoLActionKind.DISCRETE
                else None
            )
            actor = self.actor(
                batch.obs[index, timestep],
                actor_hidden,
                batch.masks[index, timestep],
                batch.physical_graphs[index, timestep],
                available,
                noise=batch.actor_noise[index, timestep],
            )
            critic = self.critic(
                batch.obs[index, timestep],
                critic_hidden,
                batch.masks[index, timestep],
                batch.physical_graphs[index, timestep],
                noise=batch.critic_noise[index, timestep],
            )
            actor_hidden, critic_hidden = actor.hidden, critic.hidden
            log_probs.append(
                self.actor.action_head.log_prob(actor.distribution, batch.actions[index, timestep])
            )
            entropies.append(self.actor.action_head.entropy(actor.distribution))
            values.append(critic.values)
            worlds.append(actor.world_prediction)
            messages.append(critic.message.detach())
            if actor.interaction_prediction is not None:
                interactions.append(actor.interaction_prediction)
        return _Unroll(
            log_probs=torch.stack(log_probs, dim=1),
            entropies=torch.stack(entropies, dim=1),
            values=torch.stack(values, dim=1),
            world_predictions=torch.stack(worlds, dim=1),
            interaction_predictions=torch.stack(interactions, dim=1) if interactions else None,
            teacher_messages=torch.stack(messages, dim=1),
        )

    def losses(
        self, batch: IWoLBatch, index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        output = self._unroll(batch, index)
        active = batch.valid[index] * batch.active_masks[index]
        denominator = active.sum().clamp_min(1.0)
        ratio = (output.log_probs - batch.old_log_probs[index]).exp()
        advantages = batch.advantages[index]
        surrogate = torch.minimum(
            ratio * advantages,
            ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages,
        )
        policy_loss = -(surrogate * active).sum() / denominator
        entropy = (output.entropies * active).sum() / denominator

        clipped_values = batch.old_values[index] + (
            output.values - batch.old_values[index]
        ).clamp(-self.clip_epsilon, self.clip_epsilon)
        normalized_returns = self.value_normalizer.normalize(batch.returns[index])
        original = huber_loss(normalized_returns - output.values, self.huber_delta)
        clipped = huber_loss(normalized_returns - clipped_values, self.huber_delta)
        value_loss = (torch.maximum(original, clipped) * active).sum() / denominator

        world_error = (output.world_predictions - batch.states[index]).square().mean(dim=-1)
        world_loss = (world_error * active).sum() / denominator
        interaction_loss = output.values.new_zeros(())
        if output.interaction_predictions is not None:
            interaction_error = (
                output.interaction_predictions - output.teacher_messages
            ).square().mean(dim=-1)
            interaction_loss = (interaction_error * active).sum() / denominator
        return policy_loss, value_loss, entropy, world_loss, interaction_loss

    def update(
        self, batch: IWoLBatch, *, epochs: int = 15, num_minibatches: int = 1,
    ) -> dict[str, float]:
        """Apply released recurrent PPO and IWoL auxiliary objectives."""
        if epochs < 1 or num_minibatches < 1:
            raise ValueError("epochs and num_minibatches must be positive")
        active = batch.valid.bool() & batch.active_masks.bool()
        valid_advantages = batch.advantages[active]
        if valid_advantages.numel() == 0:
            raise ValueError("IWoL update requires at least one active transition")
        normalized = (batch.advantages - valid_advantages.mean()) / (
            valid_advantages.std(unbiased=False) + 1e-5
        )
        batch = replace(batch, advantages=normalized)
        totals = torch.zeros(5, device=batch.obs.device)
        updates = 0
        for _ in range(epochs):
            permutation = torch.randperm(batch.obs.shape[0], device=batch.obs.device)
            for index in permutation.chunk(num_minibatches):
                return_mask = batch.valid[index].bool() & batch.active_masks[index].bool()
                valid_returns = batch.returns[index][return_mask]
                self.value_normalizer.update(valid_returns)
                policy, value, entropy, world, interaction = self.losses(batch, index)

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss = policy - self.entropy_coef * entropy + self.world_coef * world
                if self.mode is IWoLMode.IMPLICIT:
                    actor_loss = actor_loss + self.interaction_coef * interaction
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_loss_coef * value).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                totals += torch.stack(
                    (policy.detach(), value.detach(), entropy.detach(), world.detach(), interaction.detach())
                )
                updates += 1
        means = totals / updates
        return {
            "policy_loss": float(means[0]),
            "value_loss": float(means[1]),
            "entropy": float(means[2]),
            "world_loss": float(means[3]),
            "interaction_loss": float(means[4]),
            "updates": float(updates),
        }

    def graph_rate(self, graph: Tensor) -> float:
        return float(communication_rate(graph.detach()))



__all__ = [
    "IWoLActionKind",
    "IWoLAgent",
    "IWoLBatch",
    "IWoLMode",
    "IWoLRollout",
    "IWoLStep",
]
