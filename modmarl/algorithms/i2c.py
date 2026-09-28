"""Complete I2C (Individually Inferred Communication) learner.

Original paper:
Ziluo Ding, Tiejun Huang, Zongqing Lu. "Learning Individually Inferred Communication for
Multi-Agent Cooperation." Advances in Neural Information Processing Systems 33 (NeurIPS 2020).
arXiv:2006.06455.
Official source: ``PKU-AI-Edge/I2C`` revision
``3f8aab6a69ed2a46236c454bd2336c09ff6ffa36``.

Built on MADDPG with the release's parameter-shared actor, critic, message encoder, prior, and
target networks. The distinctive part is a *prior network* that lets each agent decide, per other
agent, whether communication is worth requesting: for receiver i and sender j the prior
b_i(o_i, l_ij) consumes j's relative position and predicts whether j's information is needed.
Only teammates visible under the scenario's nearest-neighbour observation rule are candidates.
When that probability
exceeds a threshold, j's raw observation is packed into the receiver's requested-message
sequence and encoded by the paper's two-layer recurrent message encoder. Communication is one-shot
per step (no message threading across the episode).

Prior-network target. The paper labels the prior by the causal effect of
j on i, I_ij = KL( P(a_i | a_{-i}, o) || P(a_i | a_{-ij}, o) ), thresholded into a binary label
and fit with cross-entropy. The centralized critic induces both conditional
action distributions by enumerating receiver actions and jointly normalizing over
receiver/sender actions before marginalizing the sender. Their KL is labelled by the
paper's running percentile threshold. The prior is trained only by this
supervised loss -- the communication gate it induces is detached from the actor/critic
objectives -- so the two learning signals stay separate. ``I2CAgent.update`` also
implements the paper's per-agent MADDPG objective and correlation regularizer.

Training follows the paper's reported two-phase protocol: a pretrained CTDE teacher supplies
causal-effect labels, then the prior is frozen while a fresh actor, message encoder, and critic
learn the communication-conditioned policy. The release's executable networks use ReLU despite
the appendix saying
LeakyReLU; this implementation follows the released code and its three-layer MLPs.
The replay retains the release's soft Gumbel action vectors and the candidate-relative-position
metadata needed to reconstruct target messages; it also retains the exact three-slot raw message
matrix used by the behaviour policy for the actor/message-encoder update. Requested messages are
packed contiguously as specified by the paper and the release's rollout path; this also avoids the
release's target-message loop bug, which repeatedly overwrites slot zero. One randomly selected
agent objective is updated at each optimization step, as in the authors' shared-variable training
loop. Generic environments without
candidate geometry may still exercise the learner with zero relative positions, but empirical
paper-fidelity claims use the included seven-agent Cooperative Navigation scenario.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.nn import build_mlp
from ..components import CentralizedMLPCritic, gumbel_policy_sample, soft_update_module


@dataclass(frozen=True)
class I2CUpdate:
    critic_loss: float
    policy_loss: float
    correlation_loss: float
    communication_rate: float


@dataclass(frozen=True)
class I2CReplayBatch:
    obs: Tensor
    actions: Tensor
    rewards: Tensor
    next_obs: Tensor
    dones: Tensor
    candidate_locations: Tensor | None = None
    candidate_mask: Tensor | None = None
    next_candidate_locations: Tensor | None = None
    next_candidate_mask: Tensor | None = None
    messages: Tensor | None = None


class I2CReplayBuffer:
    """Replay that preserves the release's soft categorical action vectors."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.candidate_locations = np.zeros((capacity, n_agents, n_agents, 2), dtype=np.float32)
        self.candidate_mask = np.zeros((capacity, n_agents, n_agents), dtype=bool)
        self.next_candidate_locations = np.zeros_like(self.candidate_locations)
        self.next_candidate_mask = np.zeros_like(self.candidate_mask)
        self.messages = np.zeros((capacity, n_agents, 3, obs_dim), dtype=np.float32)
        self.size = self.ptr = 0

    def add(
        self, obs, actions, reward, next_obs, done, *, candidate_locations=None,
        candidate_mask=None, next_candidate_locations=None, next_candidate_mask=None,
        messages=None,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        n = self.candidate_mask.shape[1]
        default_mask = ~np.eye(n, dtype=bool)
        self.candidate_locations[self.ptr] = 0.0 if candidate_locations is None else candidate_locations
        self.candidate_mask[self.ptr] = default_mask if candidate_mask is None else candidate_mask
        self.next_candidate_locations[self.ptr] = (
            0.0 if next_candidate_locations is None else next_candidate_locations
        )
        self.next_candidate_mask[self.ptr] = (
            default_mask if next_candidate_mask is None else next_candidate_mask
        )
        self.messages[self.ptr] = 0.0 if messages is None else messages
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> I2CReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        values = (
            self.obs, self.actions, self.rewards, self.next_obs, self.dones,
            self.candidate_locations, self.candidate_mask, self.next_candidate_locations,
            self.next_candidate_mask,
            self.messages,
        )
        return I2CReplayBatch(*(torch.as_tensor(value[indices], device=device) for value in values))

    def __len__(self) -> int:
        return self.size


class I2CPolicy(nn.Module):
    """Shared I2C encoder, request prior, and communication-conditioned actor.

    forward(obs) -> (logits, prior_logits):
        obs:          (batch, n_agents, obs_dim)
        logits:       (batch, n_agents, action_dim)
        prior_logits: (batch, n_agents, n_agents)   entry [b, i, j] = receiver i's request logit for sender j
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int | None = None,
        hidden_dim: int = 128,
        max_messages: int = 3,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.message_dim = obs_dim if message_dim is None else message_dim
        self.max_messages = max_messages
        self.threshold = threshold

        # The release passes zero-padded raw observation slots through a two-layer LSTM.
        self.message_encoder = nn.LSTM(
            input_size=obs_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True,
        )
        self.message_projection = nn.Linear(hidden_dim, self.message_dim)
        self.prior_net = build_mlp(obs_dim + 2, [hidden_dim, hidden_dim], 2)
        self.actor_head = build_mlp(
            obs_dim + self.message_dim, [hidden_dim, hidden_dim], action_dim,
        )

    def encode(self, obs: Tensor) -> Tensor:
        """Return the raw per-sender payload used by the I2C request/reply channel."""
        return obs

    def pack_messages(self, obs: Tensor, gate: Tensor) -> Tensor:
        """Build the release's three-slot, zero-padded raw message matrix."""
        _batch, n, obs_dim = obs.shape
        messages = []
        for receiver in range(n):
            requested = gate[:, receiver].bool()
            order = torch.argsort(requested.to(torch.int64), dim=1, descending=True, stable=True)
            packed = obs.gather(1, order.unsqueeze(-1).expand(-1, -1, obs_dim))[:, :self.max_messages]
            if packed.shape[1] < self.max_messages:
                packed = F.pad(packed, (0, 0, 0, self.max_messages - packed.shape[1]))
            positions = torch.arange(self.max_messages, device=obs.device).unsqueeze(0)
            packed = packed * (positions < requested.sum(1, keepdim=True)).unsqueeze(-1)
            messages.append(packed)
        return torch.stack(messages, dim=1)

    def encode_messages(self, messages: Tensor) -> Tensor:
        """Encode raw message matrices -> one vector per receiver."""
        batch, n, slots, obs_dim = messages.shape
        encoded, _ = self.message_encoder(messages.reshape(batch * n, slots, obs_dim))
        projected = self.message_projection(encoded[:, -1])
        return projected.view(batch, n, self.message_dim)

    def aggregate(self, obs: Tensor, gate: Tensor) -> Tensor:
        """Pack requested observations first, zero-pad, and recurrently encode them."""
        return self.encode_messages(self.pack_messages(obs, gate))

    def prior(self, obs: Tensor, candidate_locations: Tensor) -> Tensor:
        """Probability log-odds that receiver i requests candidate sender j."""
        own_obs = obs.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        class_logits = self.prior_net(torch.cat([own_obs, candidate_locations], dim=-1))
        return class_logits[..., 0] - class_logits[..., 1]

    def act(self, obs: Tensor, message: Tensor) -> Tensor:
        """Action logits from an agent's own observation and its aggregated message."""
        return self.actor_head(torch.cat([obs, message], dim=-1))

    def act_from_messages(self, obs: Tensor, messages: Tensor) -> Tensor:
        """Apply the shared actor to replayed raw message matrices."""
        return self.act(obs, self.encode_messages(messages))

    def _gate(self, prior_logits: Tensor, candidate_mask: Tensor | None = None) -> Tensor:
        """Hard communication gate: request j iff b_i(o_i, d_j) > threshold; never self. The
        gate is detached -- the prior is trained by its own supervised loss, not the actor's."""
        n = self.n_agents
        gate = (torch.sigmoid(prior_logits) > self.threshold).to(prior_logits.dtype)
        no_self = ~torch.eye(n, device=prior_logits.device, dtype=torch.bool)
        valid = no_self if candidate_mask is None else no_self & candidate_mask.bool()
        return (gate * valid.to(gate.dtype)).detach()

    def forward(
        self, obs: Tensor, candidate_locations: Tensor | None = None,
        candidate_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, n, _ = obs.shape
        if candidate_locations is None:
            candidate_locations = obs.new_zeros(batch, n, n, 2)
        prior_logits = self.prior(obs, candidate_locations) # (batch, n, n)
        gate = self._gate(prior_logits, candidate_mask)     # (batch, n, n)
        message = self.aggregate(obs, gate)                 # recurrent encoding of requested raw observations
        logits = self.act(obs, message)                     # (batch, n, action)
        return logits, prior_logits

    def sample(
        self,
        obs: Tensor,
        candidate_locations: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        temperature: float = 1.0,
        hard: bool = False,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, prior_logits = self(obs, candidate_locations, candidate_mask)
        one_hot, action_idx, logits = gumbel_policy_sample(
            logits, action_dim=self.action_dim, temperature=temperature, hard=hard, deterministic=deterministic,
        )
        return one_hot, action_idx, logits, prior_logits

    def sample_from_messages(
        self, obs: Tensor, messages: Tensor, temperature: float = 1.0,
        hard: bool = False, deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.act_from_messages(obs, messages)
        action, action_idx, logits = gumbel_policy_sample(
            logits, action_dim=self.action_dim, temperature=temperature,
            hard=hard, deterministic=deterministic,
        )
        return action, action_idx, logits

    def actor_parameters(self):
        """Parameters trained by the MADDPG policy objective (encoder + actor head)."""
        return chain(
            self.message_encoder.parameters(), self.message_projection.parameters(),
            self.actor_head.parameters(),
        )

    def prior_parameters(self):
        """Parameters trained by the supervised prior objective."""
        return self.prior_net.parameters()


class I2CAgent(nn.Module):
    """I2C's parameter-shared decentralized policy and centralized critic."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int | None = None,
        hidden_dim: int = 128,
        max_messages: int = 3,
        threshold: float = 0.5,
        policy_lr: float = 1e-2,
        critic_lr: float = 1e-2,
        prior_lr: float = 1e-2,
        influence_temperature: float = 1.0,
        correlation_temperature: float = 8.0,
        influence_percentile: float = 80.0,
        correlation_coefficient: float = 1e-2,
    ) -> None:
        super().__init__()
        self.policy = I2CPolicy(
            n_agents, obs_dim, action_dim,
            message_dim=message_dim, hidden_dim=hidden_dim, max_messages=max_messages,
            threshold=threshold,
        )
        self.critic = CentralizedMLPCritic(
            n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        self.target_policy = copy.deepcopy(self.policy)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_policy.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.influence_temperature = influence_temperature
        self.correlation_temperature = correlation_temperature
        self.influence_percentile = influence_percentile
        self.correlation_coefficient = correlation_coefficient
        self.policy_optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.prior_optimizer = torch.optim.Adam(self.policy.prior_parameters(), lr=prior_lr)

    def soft_update(self, tau: float) -> None:
        soft_update_module(self.target_policy, self.policy, tau)
        soft_update_module(self.target_critic, self.critic, tau)

    def conditional_action_distribution(self, obs: Tensor, actions: Tensor, receiver: int) -> Tensor:
        """Paper Eq. 2: ``P(a_i | a_-i, o)`` induced by the critic."""
        batch, n, action_dim = actions.shape
        candidates = actions[:, None].expand(batch, action_dim, n, action_dim).clone()
        candidates[:, :, receiver] = torch.eye(action_dim, device=obs.device, dtype=obs.dtype)
        tiled_obs = obs[:, None].expand(-1, action_dim, -1, -1)
        q = self.critic(
            tiled_obs.reshape(-1, n, obs.shape[-1]), candidates.reshape(-1, n, action_dim),
        ).view(batch, action_dim)
        return torch.softmax(self.influence_temperature * q, dim=-1)

    def correlation_action_distribution(self, obs: Tensor, actions: Tensor, receiver: int) -> Tensor:
        """Released correlation target induced by the centralized critic."""
        batch, n, action_dim = actions.shape
        candidates = actions[:, None].expand(batch, action_dim, n, action_dim).clone()
        candidates[:, :, receiver] = torch.eye(action_dim, device=obs.device, dtype=obs.dtype)
        tiled_obs = obs[:, None].expand(-1, action_dim, -1, -1)
        q = self.critic(
            tiled_obs.reshape(-1, n, obs.shape[-1]), candidates.reshape(-1, n, action_dim),
        ).view(batch, action_dim)
        centered = q - q.mean(dim=-1, keepdim=True)
        scale = centered.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.softmax(self.correlation_temperature * centered / scale, dim=-1)

    @torch.no_grad()
    def causal_influence(
        self, obs: Tensor, actions: Tensor, *, receiver: int | None = None,
        candidate_mask: Tensor | None = None,
    ) -> Tensor:
        """Paper Eqs. 1--3 for every directed receiver/sender pair."""
        batch, n, action_dim = actions.shape
        influences = obs.new_zeros(batch, n, n)
        eye = torch.eye(action_dim, device=obs.device, dtype=obs.dtype)
        receivers = range(n) if receiver is None else (receiver,)
        for current_receiver in receivers:
            conditional = self.conditional_action_distribution(obs, actions, current_receiver)
            for sender in range(n):
                if sender == current_receiver:
                    continue
                if candidate_mask is not None and not bool(candidate_mask[:, current_receiver, sender].any()):
                    continue
                joint = actions[:, None, None].expand(batch, action_dim, action_dim, n, action_dim).clone()
                joint[:, :, :, current_receiver] = eye.view(1, action_dim, 1, action_dim)
                joint[:, :, :, sender] = eye.view(1, 1, action_dim, action_dim)
                tiled_obs = obs[:, None, None].expand(-1, action_dim, action_dim, -1, -1)
                q = self.critic(
                    tiled_obs.reshape(-1, n, obs.shape[-1]), joint.reshape(-1, n, action_dim),
                ).view(batch, action_dim, action_dim)
                joint_probability = torch.softmax(
                    self.influence_temperature * q.flatten(1), dim=-1,
                ).view_as(q)
                marginal = joint_probability.sum(dim=2).clamp_min(1e-8)
                influences[:, current_receiver, sender] = (
                    conditional * (conditional.clamp_min(1e-8).log() - marginal.log())
                ).sum(dim=-1)
        return influences

    def update(
        self, batch, *, gamma: float = 0.95, tau: float = 0.01,
        max_grad_norm: float = 0.5, receiver: int | None = None,
        full_communication: bool = False,
    ) -> I2CUpdate:
        """Update one randomly selected actor objective and the shared critic."""
        replay_actions = batch.actions.to(batch.obs.dtype)
        if receiver is None:
            receiver = int(torch.randint(self.policy.n_agents, (), device=batch.obs.device))
        candidate_locations = batch.candidate_locations
        candidate_mask = batch.candidate_mask
        if candidate_locations is None:
            candidate_locations = batch.obs.new_zeros(
                batch.obs.shape[0], self.policy.n_agents, self.policy.n_agents, 2,
            )

        with torch.no_grad():
            if full_communication:
                next_gate = batch.next_candidate_mask.to(batch.next_obs.dtype)
                next_messages = self.target_policy.pack_messages(batch.next_obs, next_gate)
                next_actions, _, _ = self.target_policy.sample_from_messages(
                    batch.next_obs, next_messages, hard=False,
                )
            else:
                next_actions, _, _, _ = self.target_policy.sample(
                    batch.next_obs, batch.next_candidate_locations,
                    batch.next_candidate_mask, hard=False,
                )
            targets = batch.rewards + gamma * (1.0 - batch.dones) * self.target_critic(
                batch.next_obs, next_actions,
            )
        critic_loss = F.mse_loss(self.critic(batch.obs, replay_actions), targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()

        if batch.messages is None:
            prior_logits_for_messages = self.policy.prior(batch.obs, candidate_locations)
            messages = self.policy.pack_messages(
                batch.obs, self.policy._gate(prior_logits_for_messages, candidate_mask),
            )
        else:
            messages = batch.messages
        policy_actions, _, _ = self.policy.sample_from_messages(
            batch.obs, messages, hard=False,
        )
        counterfactual = replay_actions.clone()
        counterfactual[:, receiver] = policy_actions[:, receiver]
        actor_loss = -self.critic(batch.obs, counterfactual).mean()
        with torch.no_grad():
            desired = self.correlation_action_distribution(batch.obs, replay_actions, receiver)
        actual = policy_actions[:, receiver].clamp_min(1e-8)
        correlation_loss = (
            desired * (desired.clamp_min(1e-8).log() - actual.log())
        ).sum(-1).mean()
        policy_loss = actor_loss + self.correlation_coefficient * correlation_loss
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.actor_parameters()), max_grad_norm)
        self.policy_optimizer.step()

        prior_logits = self.policy.prior(batch.obs, candidate_locations)
        valid = ~torch.eye(
            self.policy.n_agents, device=batch.obs.device, dtype=torch.bool,
        ).unsqueeze(0).expand_as(prior_logits)
        if candidate_mask is not None:
            valid = valid & candidate_mask.bool()
        gate_rate = (
            self.policy._gate(prior_logits, candidate_mask)[valid].mean()
            if valid.any() else batch.obs.new_zeros(())
        )

        self.soft_update(tau)
        return I2CUpdate(
            critic_loss=float(critic_loss.detach()), policy_loss=float(policy_loss.detach()),
            correlation_loss=float(correlation_loss.detach()), communication_rate=float(gate_rate),
        )

    def fit_prior(
        self, obs: Tensor, candidate_locations: Tensor, labels: Tensor, *,
        steps: int, batch_size: int, max_grad_norm: float = 0.5,
    ) -> float:
        """Fit the prior from a balanced causal-effect dataset."""
        positive = torch.nonzero(labels.bool(), as_tuple=False).flatten()
        negative = torch.nonzero(~labels.bool(), as_tuple=False).flatten()
        if positive.numel() == 0 or negative.numel() == 0:
            raise ValueError("causal dataset must contain positive and negative examples")
        losses = []
        half = max(1, batch_size // 2)
        for _ in range(steps):
            indices = torch.cat([
                positive[torch.randint(positive.numel(), (half,), device=obs.device)],
                negative[torch.randint(negative.numel(), (half,), device=obs.device)],
            ])
            logits = self.policy.prior_net(
                torch.cat([obs[indices], candidate_locations[indices]], dim=-1),
            )
            loss = F.cross_entropy(logits, 1 - labels[indices].long())
            self.prior_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.prior_parameters()), max_grad_norm)
            self.prior_optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses)) if losses else 0.0

    def load_frozen_prior(self, source: I2CAgent) -> None:
        """Copy a trained prior into both networks and freeze it for phase two."""
        state = source.policy.prior_net.state_dict()
        self.policy.prior_net.load_state_dict(state)
        self.target_policy.prior_net.load_state_dict(state)
        self.policy.prior_net.requires_grad_(False)
        self.target_policy.prior_net.requires_grad_(False)


__all__ = ["I2CAgent", "I2CPolicy", "I2CReplayBatch", "I2CReplayBuffer", "I2CUpdate"]
