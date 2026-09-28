"""The discrete Multi-Agent Transformer: backbone and on-policy PPO learner.

Model: an unmasked observation encoder produces one value and representation per
agent; a causal decoder maps a start token and preceding one-hot actions to the
joint categorical policy.  ``MATRollout`` stores fresh joint transitions and closes
each episode with per-agent GAE, and ``MATAgent`` jointly optimizes the encoder value
head and causal decoder with the released PPO objective.  Invariants: tensors retain
``(batch, agent, feature)`` ordering, teacher-forced and autoregressive
log-probabilities agree, agent ``i`` can depend only on actions ``< i``, no replayed
rollout is needed, value predictions stay in ValueNorm space, and time-limit
truncations bootstrap.  Interface: collect with ``act``/``MATRollout`` and consume a
completed ``MATBatch`` with ``update``.
Why: MAT's advantage decomposition is realized by the action ordering, so the decoder
contract is kept explicit, and MAT trains action sequences as one joint sample, which
ordinary flattened per-agent PPO buffers cannot represent without losing causal context.

Paper: Wen et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling
Problem," NeurIPS 2022 (arXiv:2205.14953).  Architecture and defaults follow
``PKU-MARL/Multi-Agent-Transformer`` revision
``be3ff49c8264d454c1fe2c41582aa2bfc98498c8``: one 64-wide, one-head block by default,
orthogonal initialization, GELU feed-forwards, the released shifted discrete-action
representation, Adam 5e-4 (epsilon 1e-5), 15 PPO epochs, one minibatch, clip 0.2,
entropy 0.01, gamma 0.99, GAE 0.95, ValueNorm, clipped Huber value loss (delta 10),
and gradient norm 10.  The release overwrites every supplied global state with a
hard-coded 37-vector; modMARL instead makes observation encoding the default and
supports an explicitly sized state only when ``encode_state=True``.
Validation uses the dependency-free three-agent cooperative navigation task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mappo import ValueNorm, huber_loss


def _orthogonal(linear: nn.Linear, *, activation: bool = False) -> nn.Linear:
    gain = nn.init.calculate_gain("relu") if activation else 0.01
    nn.init.orthogonal_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class MATSelfAttention(nn.Module):
    """Released multi-head dot-product attention with an optional causal mask."""

    def __init__(self, embedding_dim: int, n_heads: int, n_agents: int, *, causal: bool) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.causal = causal
        self.key = _orthogonal(nn.Linear(embedding_dim, embedding_dim))
        self.query = _orthogonal(nn.Linear(embedding_dim, embedding_dim))
        self.value = _orthogonal(nn.Linear(embedding_dim, embedding_dim))
        self.output = _orthogonal(nn.Linear(embedding_dim, embedding_dim))
        self.register_buffer("causal_mask", torch.tril(torch.ones(n_agents, n_agents, dtype=torch.bool)))

    def forward(self, key: Tensor, value: Tensor, query: Tensor) -> Tensor:
        """Attend over agent sequences, each shaped ``(batch, agents, embedding)``."""
        batch, query_length, width = query.shape
        key_length = key.shape[1]
        head_width = width // self.n_heads

        q = self.query(query).view(batch, query_length, self.n_heads, head_width).transpose(1, 2)
        k = self.key(key).view(batch, key_length, self.n_heads, head_width).transpose(1, 2)
        v = self.value(value).view(batch, key_length, self.n_heads, head_width).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)
        if self.causal:
            mask = self.causal_mask[:query_length, :key_length]
            scores = scores.masked_fill(~mask, -torch.inf)
        weights = scores.softmax(dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, query_length, width)
        return self.output(attended)


class _EncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, n_agents: int) -> None:
        super().__init__()
        self.attention = MATSelfAttention(embedding_dim, n_heads, n_agents, causal=False)
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
            nn.GELU(),
            _orthogonal(nn.Linear(embedding_dim, embedding_dim)),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.attention_norm(inputs + self.attention(inputs, inputs, inputs))
        return self.output_norm(hidden + self.feed_forward(hidden))


class _DecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, n_agents: int) -> None:
        super().__init__()
        self.action_attention = MATSelfAttention(embedding_dim, n_heads, n_agents, causal=True)
        self.action_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = MATSelfAttention(embedding_dim, n_heads, n_agents, causal=True)
        self.cross_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
            nn.GELU(),
            _orthogonal(nn.Linear(embedding_dim, embedding_dim)),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, actions: Tensor, observation_representation: Tensor) -> Tensor:
        hidden = self.action_norm(actions + self.action_attention(actions, actions, actions))
        hidden = self.cross_norm(
            observation_representation
            + self.cross_attention(hidden, hidden, observation_representation)
        )
        return self.output_norm(hidden + self.feed_forward(hidden))


class MATBackbone(nn.Module):
    """MAT observation encoder, local value heads, and causal action decoder."""

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
    ) -> None:
        super().__init__()
        if n_agents < 1 or action_dim < 2:
            raise ValueError("MAT requires at least one agent and two discrete actions")
        if encode_state and state_dim is None:
            raise ValueError("state_dim is required when encode_state=True")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.encode_state = encode_state

        input_dim = state_dim if encode_state else obs_dim
        assert input_dim is not None
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            _orthogonal(nn.Linear(input_dim, embedding_dim), activation=True),
            nn.GELU(),
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.encoder_blocks = nn.ModuleList(
            [_EncoderBlock(embedding_dim, n_heads, n_agents) for _ in range(n_blocks)]
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
            [_DecoderBlock(embedding_dim, n_heads, n_agents) for _ in range(n_blocks)]
        )
        self.action_head = nn.Sequential(
            _orthogonal(nn.Linear(embedding_dim, embedding_dim), activation=True),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            _orthogonal(nn.Linear(embedding_dim, action_dim)),
        )

    def _validate_obs(self, obs: Tensor) -> None:
        if obs.ndim != 3 or obs.shape[1:] != (self.n_agents, self.obs_dim):
            raise ValueError(
                f"obs must have shape (batch, {self.n_agents}, {self.obs_dim}), got {tuple(obs.shape)}"
            )

    def encode(self, obs: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return normalized local values and agent representations."""
        self._validate_obs(obs)
        if self.encode_state:
            if state is None or state.shape[:2] != obs.shape[:2] or state.shape[-1] != self.state_dim:
                raise ValueError("state must match (batch, agents, state_dim)")
            inputs = state
        else:
            inputs = obs
        representation = self.encoder_norm(self.input_encoder(inputs))
        for block in self.encoder_blocks:
            representation = block(representation)
        return self.value_head(representation).squeeze(-1), representation

    def shifted_actions(self, actions: Tensor) -> Tensor:
        """Build the released start-token/previous-one-hot decoder input."""
        if actions.ndim != 2 or actions.shape[1] != self.n_agents:
            raise ValueError(f"actions must have shape (batch, {self.n_agents})")
        shifted = torch.zeros(
            actions.shape[0], self.n_agents, self.action_dim + 1,
            dtype=torch.float32, device=actions.device,
        )
        shifted[:, 0, 0] = 1.0
        if self.n_agents > 1:
            shifted[:, 1:, 1:] = F.one_hot(
                actions[:, :-1].long(), num_classes=self.action_dim,
            ).to(shifted.dtype)
        return shifted

    def decode(self, shifted_actions: Tensor, representation: Tensor) -> Tensor:
        """Return categorical logits for every agent in parallel."""
        hidden = self.decoder_norm(self.action_encoder(shifted_actions))
        for block in self.decoder_blocks:
            hidden = block(hidden, representation)
        return self.action_head(hidden)

    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
        available_actions: Tensor | None = None,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate stored actions; return ``(log_probs, values, entropies)`` as ``(B,N)``."""
        values, representation = self.encode(obs, state)
        logits = self.decode(self.shifted_actions(actions), representation)
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
        """Autoregressively return actions, their log-probabilities, and local values."""
        values, representation = self.encode(obs, state)
        batch = obs.shape[0]
        shifted = torch.zeros(
            batch, self.n_agents, self.action_dim + 1, dtype=obs.dtype, device=obs.device,
        )
        shifted[:, 0, 0] = 1.0
        actions = torch.zeros(batch, self.n_agents, dtype=torch.long, device=obs.device)
        log_probs = torch.zeros(batch, self.n_agents, dtype=obs.dtype, device=obs.device)
        for agent in range(self.n_agents):
            logits = self.decode(shifted, representation)[:, agent]
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
                    action, num_classes=self.action_dim,
                ).to(shifted.dtype)
        return actions, log_probs, values

    def values(self, obs: Tensor, state: Tensor | None = None) -> Tensor:
        """Return normalized local value predictions shaped ``(batch, agents)``."""
        return self.encode(obs, state)[0]


@dataclass(frozen=True)
class MATBatch:
    """Joint on-policy transitions; leading dimensions are ``(samples, agents)``."""

    obs: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    active_masks: Tensor
    available_actions: Tensor
    states: Tensor | None = None


class MATRollout:
    """Fresh complete episodes with every input needed for parallel decoding."""

    def __init__(self, learner: _SequencePPOAgent) -> None:
        self.learner = learner
        self._episodes: list[MATBatch] = []
        self._obs: list[Tensor] = []
        self._actions: list[Tensor] = []
        self._log_probs: list[Tensor] = []
        self._values: list[Tensor] = []
        self._rewards: list[Tensor] = []
        self._active_masks: list[Tensor] = []
        self._available_actions: list[Tensor] = []
        self._states: list[Tensor] = []

    def add(
        self,
        *,
        obs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        team_reward: float,
        active_masks: Tensor | None = None,
        available_actions: Tensor | None = None,
        state: Tensor | None = None,
    ) -> None:
        """Append one joint step; every supplied tensor has a leading agent axis."""
        if active_masks is None:
            active_masks = torch.ones_like(values)
        if available_actions is None:
            available_actions = torch.ones(
                self.learner.n_agents,
                self.learner.action_dim,
                dtype=torch.bool,
                device=actions.device,
            )
        self._obs.append(obs.detach())
        self._actions.append(actions.detach())
        self._log_probs.append(log_probs.detach())
        self._values.append(values.detach())
        self._rewards.append(torch.full_like(values, float(team_reward)))
        self._active_masks.append(active_masks.detach())
        self._available_actions.append(available_actions.detach())
        if state is not None:
            self._states.append(state.detach())
        elif self.learner.backbone.encode_state:
            raise ValueError("state is required by an encode_state MAT learner")

    def finish_episode(self, bootstrap_value: Tensor, final_mask: Tensor) -> None:
        """Close the current episode; ``final_mask`` is zero only for true termination."""
        if not self._rewards:
            raise ValueError("cannot finish an empty MAT episode")
        values = torch.stack(self._values)
        rewards = torch.stack(self._rewards)
        masks = torch.ones_like(rewards)
        masks[-1] = final_mask
        advantages, returns = self.learner.compute_gae(rewards, values, bootstrap_value, masks)
        states = torch.stack(self._states) if self._states else None
        self._episodes.append(
            MATBatch(
                obs=torch.stack(self._obs),
                actions=torch.stack(self._actions),
                old_log_probs=torch.stack(self._log_probs),
                old_values=values,
                advantages=advantages,
                returns=returns,
                active_masks=torch.stack(self._active_masks),
                available_actions=torch.stack(self._available_actions),
                states=states,
            )
        )
        self._obs.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._active_masks.clear()
        self._available_actions.clear()
        self._states.clear()

    def batch(self) -> MATBatch:
        """Concatenate completed episodes without flattening the agent sequence."""
        if self._rewards:
            raise RuntimeError("finish the current MAT episode before requesting a batch")
        if not self._episodes:
            raise RuntimeError("the MAT rollout contains no completed episodes")
        states = None
        if self._episodes[0].states is not None:
            if any(episode.states is None for episode in self._episodes):
                raise RuntimeError("MAT rollout mixes state-encoded and observation-encoded episodes")
            states = torch.cat([episode.states for episode in self._episodes if episode.states is not None])
        return MATBatch(
            obs=torch.cat([episode.obs for episode in self._episodes]),
            actions=torch.cat([episode.actions for episode in self._episodes]),
            old_log_probs=torch.cat([episode.old_log_probs for episode in self._episodes]),
            old_values=torch.cat([episode.old_values for episode in self._episodes]),
            advantages=torch.cat([episode.advantages for episode in self._episodes]),
            returns=torch.cat([episode.returns for episode in self._episodes]),
            active_masks=torch.cat([episode.active_masks for episode in self._episodes]),
            available_actions=torch.cat([episode.available_actions for episode in self._episodes]),
            states=states,
        )


class _SequencePPOAgent(nn.Module):
    """Shared MAT/CommFormer PPO mechanics; subclasses own backbone and optimizer cadence."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        n_agents: int,
        action_dim: int,
        learning_rate: float,
        clip_epsilon: float,
        entropy_coef: float,
        value_loss_coef: float,
        max_grad_norm: float,
        huber_delta: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.value_normalizer = ValueNorm()
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.huber_delta = huber_delta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(), lr=learning_rate, eps=1e-5, weight_decay=0.0,
        )

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        available_actions: Tensor | None = None,
        *,
        state: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return joint actions, behavior log-probabilities, and normalized values."""
        squeeze = obs.ndim == 2
        if squeeze:
            obs = obs.unsqueeze(0)
            available_actions = None if available_actions is None else available_actions.unsqueeze(0)
            state = None if state is None else state.unsqueeze(0)
        actions, log_probs, values = self.backbone.act(
            obs, available_actions, state=state, deterministic=deterministic,
        )
        if squeeze:
            return actions[0], log_probs[0], values[0]
        return actions, log_probs, values

    @torch.no_grad()
    def values(self, obs: Tensor, state: Tensor | None = None) -> Tensor:
        squeeze = obs.ndim == 2
        if squeeze:
            obs = obs.unsqueeze(0)
            state = None if state is None else state.unsqueeze(0)
        values = self.backbone.values(obs, state)
        return values[0] if squeeze else values

    def compute_gae(
        self,
        rewards: Tensor,
        normalized_values: Tensor,
        bootstrap_value: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute per-agent GAE using denormalized critic predictions."""
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

    def _evaluate(
        self, batch: MATBatch, index: Tensor, training_step: int, total_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del training_step, total_steps
        states = None if batch.states is None else batch.states[index]
        return self.backbone(
            batch.obs[index], batch.actions[index], batch.available_actions[index], states,
        )

    def losses(
        self,
        batch: MATBatch,
        index: Tensor,
        *,
        training_step: int = 0,
        total_steps: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Released clipped PPO, clipped ValueNorm-Huber, and entropy objectives."""
        log_probs, values, entropies = self._evaluate(batch, index, training_step, total_steps)
        active = batch.active_masks[index]
        denominator = active.sum().clamp_min(1.0)
        ratio = (log_probs - batch.old_log_probs[index]).exp()
        surrogate = torch.minimum(
            ratio * batch.advantages[index],
            ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * batch.advantages[index],
        )
        policy_loss = -(surrogate * active).sum() / denominator
        entropy = (entropies * active).sum() / denominator

        old_values = batch.old_values[index]
        clipped_values = old_values + (values - old_values).clamp(
            -self.clip_epsilon, self.clip_epsilon,
        )
        normalized_returns = self.value_normalizer.normalize(batch.returns[index])
        original = huber_loss(normalized_returns - values, self.huber_delta)
        clipped = huber_loss(normalized_returns - clipped_values, self.huber_delta)
        value_loss = (torch.maximum(original, clipped) * active).sum() / denominator
        return policy_loss, value_loss, entropy

    def _optimize(self, loss: Tensor, epoch: int, training_step: int, total_steps: int) -> str:
        del epoch, training_step, total_steps
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.backbone.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return "model"

    def update(
        self,
        batch: MATBatch,
        *,
        epochs: int = 15,
        num_minibatches: int = 1,
        training_step: int = 0,
        total_steps: int = 0,
    ) -> dict[str, float]:
        """Consume one fresh joint rollout with paper-wide advantage normalization."""
        active = batch.active_masks.bool()
        valid_advantages = batch.advantages[active]
        normalized = (
            (batch.advantages - valid_advantages.mean())
            / (valid_advantages.std(unbiased=False) + 1e-5)
        )
        batch = MATBatch(**{**batch.__dict__, "advantages": normalized})
        sample_count = batch.obs.shape[0]
        if sample_count < num_minibatches:
            raise ValueError("MAT needs at least one joint sample per minibatch")
        totals = torch.zeros(3, device=batch.obs.device)
        update_counts = {"model": 0, "edge": 0}
        updates = 0
        for epoch in range(epochs):
            permutation = torch.randperm(sample_count, device=batch.obs.device)
            for index in permutation.chunk(num_minibatches):
                active_index = batch.active_masks[index].bool()
                self.value_normalizer.update(batch.returns[index][active_index])
                policy_loss, value_loss, entropy = self.losses(
                    batch,
                    index,
                    training_step=training_step,
                    total_steps=total_steps,
                )
                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                owner = self._optimize(loss, epoch, training_step, total_steps)
                update_counts[owner] += 1
                totals += torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach()))
                updates += 1
        means = totals / updates
        return {
            "policy_loss": float(means[0]),
            "value_loss": float(means[1]),
            "entropy": float(means[2]),
            "model_updates": float(update_counts["model"]),
            "edge_updates": float(update_counts["edge"]),
        }


class MATAgent(_SequencePPOAgent):
    """Complete discrete MAT learner with released architecture and PPO defaults."""

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
        learning_rate: float = 5e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 10.0,
        huber_delta: float = 10.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        backbone = MATBackbone(
            n_agents,
            obs_dim,
            action_dim,
            state_dim=state_dim,
            encode_state=encode_state,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
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

__all__ = ["MATAgent", "MATBackbone", "MATBatch", "MATRollout", "MATSelfAttention"]
