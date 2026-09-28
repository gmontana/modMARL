"""Intention Sharing with paper-faithful per-agent MADDPG learners.

Model: each agent owns an independent actor, centralized critic, action predictor,
observation predictor, and temporal-attention message generator. The actor consumes
its local observation and the previous messages from all agents; the critic consumes
the joint observation and joint action during centralized training.
Invariants: agent parameters are never shared; the first imagined pair contains the
executed action; action prediction excludes the predicting agent; outgoing messages
remain connected to every receiver's next-step policy loss. Tensors use
``(batch, agents, feature)`` ordering at the learner boundary.
Interface: ``IntentionSharingLearner`` owns replay, optimization, target networks,
and joint sampling. ``IntentionSharingPolicy`` exposes one agent's ITGM and attention
module for inspection.
Why: the ICLR 2021 paper (OpenReview ``qpsl2dR9twy``) specifies one actor and one
critic per agent in Equations 9--11. The authors' unreleased reference code (private
Bitbucket repository ``intention_sharing``, revision
``e5d4527d7f7fecf36a1e9fe60969352eb07cd691``; not publicly accessible) instead uses one
shared "keeper" and rolls its observation model through an encoded latent. Those shortcuts are not
reproduced: Equations 3 and 13 require a residual in observation space, and the
paper's independently indexed parameters govern ownership.

Defaults follow Table 2 for Cooperative Navigation: imagination horizon 5, message
and attention dimension 3, discount 0.99, Adam at 5e-4 for every network, minibatch
128, replay 2e5, and two 128-unit ReLU hidden layers. The recovered runner's
one-update-per-100-environment-steps cadence is retained by the training example;
the paper does not report a different update interval.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from ..common.nn import build_mlp
from ..components import CentralizedMLPCritic, gumbel_policy_sample, soft_update_module


@dataclass(frozen=True)
class IntentionSharingConfig:
    """Cooperative Navigation settings reported in paper Table 2."""

    gamma: float = 0.99
    tau: float = 0.01
    learning_rate: float = 5e-4
    model_loss_weight: float = 0.1
    policy_regularization: float = 1e-3
    gradient_clip: float = 0.5
    batch_size: int = 128
    replay_capacity: int = 200_000
    minimum_replay_size: int = 1_000


@dataclass(frozen=True)
class IntentionSharingUpdate:
    """Scalar diagnostics from one joint MADDPG update."""

    critic_loss: float
    policy_loss: float
    model_loss: float


@dataclass(frozen=True)
class _ReplayBatch:
    obs: Tensor
    actions: Tensor
    rewards: Tensor
    next_obs: Tensor
    dones: Tensor
    messages_seen: Tensor
    messages_written: Tensor


class _ReplayBuffer:
    """Joint replay preserving per-agent rewards and the recurrent message state."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        message_dim: int,
    ) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)
        self.messages_seen = np.zeros((capacity, n_agents, message_dim), dtype=np.float32)
        self.messages_written = np.zeros_like(self.messages_seen)
        self.size = 0
        self.position = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
        messages_seen: np.ndarray,
        messages_written: np.ndarray,
    ) -> None:
        self.obs[self.position] = obs
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_obs[self.position] = next_obs
        self.dones[self.position] = dones
        self.messages_seen[self.position] = messages_seen
        self.messages_written[self.position] = messages_written
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> _ReplayBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return _ReplayBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
            messages_seen=torch.as_tensor(self.messages_seen[indices], device=device),
            messages_written=torch.as_tensor(self.messages_written[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


class IntentionSharingPolicy(nn.Module):
    """One agent's actor, imagined-trajectory model, and message compressor."""

    def __init__(
        self,
        agent_index: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int = 3,
        hidden_dim: int = 128,
        imagination_horizon: int = 5,
    ) -> None:
        super().__init__()
        if not 0 <= agent_index < n_agents:
            raise ValueError("agent_index must identify one of n_agents")
        if imagination_horizon < 1:
            raise ValueError("imagination_horizon must be positive")
        self.agent_index = agent_index
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.imagination_horizon = imagination_horizon
        context_dim = n_agents * message_dim
        other_action_dim = (n_agents - 1) * action_dim

        self.policy_head = build_mlp(
            obs_dim + context_dim, [hidden_dim, hidden_dim], action_dim,
        )
        self.action_model = build_mlp(
            obs_dim, [hidden_dim, hidden_dim], other_action_dim,
        )
        self.dynamics_model = build_mlp(
            obs_dim + action_dim + other_action_dim,
            [hidden_dim, hidden_dim],
            obs_dim,
        )
        trajectory_dim = obs_dim + action_dim
        self.message_query = nn.Linear(context_dim, message_dim, bias=False)
        self.trajectory_key = nn.Linear(trajectory_dim, message_dim, bias=False)
        self.trajectory_value = nn.Linear(trajectory_dim, message_dim, bias=False)

    def _context(self, incoming: Tensor) -> Tensor:
        if incoming.ndim != 3 or incoming.shape[1:] != (self.n_agents, self.message_dim):
            raise ValueError("incoming must have shape (batch, n_agents, message_dim)")
        return incoming.flatten(1)

    def imagine(
        self,
        obs: Tensor,
        incoming: Tensor,
        first_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Roll Equation 3 for one sender in observation space.

        ``obs`` is ``(B, obs_dim)`` and ``incoming`` is ``(B, N, message_dim)``.
        The returned trajectory is ``(B, H, obs_dim + action_dim)``. When supplied,
        ``first_action`` is the executed one-hot action in the true pair ``tau_t``.
        """
        context = self._context(incoming)
        current = obs
        trajectory: list[Tensor] = []
        first_other_logits = None
        first_predicted_obs = None
        for step in range(self.imagination_horizon):
            own_action = torch.softmax(
                self.policy_head(torch.cat([current, context], dim=-1)), dim=-1,
            )
            if step == 0 and first_action is not None:
                own_action = first_action
            other_logits = self.action_model(current).view(
                current.shape[0], self.n_agents - 1, self.action_dim,
            )
            predicted_other_actions = torch.softmax(other_logits, dim=-1).flatten(1)
            trajectory.append(torch.cat([current, own_action], dim=-1))
            predicted_obs = current + self.dynamics_model(
                torch.cat([current, own_action, predicted_other_actions], dim=-1),
            )
            if step == 0:
                first_other_logits = other_logits
                first_predicted_obs = predicted_obs
            current = predicted_obs
        assert first_other_logits is not None and first_predicted_obs is not None
        return torch.stack(trajectory, dim=1), first_other_logits, first_predicted_obs

    def messages(
        self,
        obs: Tensor,
        incoming: Tensor,
        first_action: Tensor,
        *,
        detach_trajectory: bool = False,
    ) -> Tensor:
        """Compress the H imagined pairs with paper Equation 7 attention."""
        trajectory, _, _ = self.imagine(obs, incoming, first_action)
        if detach_trajectory:
            # Equation 11 updates W_Q/W_K/W_V; predictors use Equations 12--13.
            trajectory = trajectory.detach()
        query = self.message_query(self._context(incoming))
        keys = self.trajectory_key(trajectory)
        values = self.trajectory_value(trajectory)
        scores = torch.einsum("bd,btd->bt", query, keys) * self.message_dim**-0.5
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", weights, values)

    def forward(
        self,
        obs: Tensor,
        incoming: Tensor,
        first_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return action logits and the outgoing intention for one agent."""
        logits = self.policy_head(torch.cat([obs, self._context(incoming)], dim=-1))
        if first_action is None:
            first_action = torch.softmax(logits, dim=-1)
        return logits, self.messages(obs, incoming, first_action)

    def _other_actions(self, actions: Tensor) -> Tensor:
        """Select the executed ``a_-i`` labels required by Equation 12."""
        keep = torch.arange(self.n_agents, device=actions.device) != self.agent_index
        return actions[:, keep]

    def model_loss(
        self,
        obs: Tensor,
        next_obs: Tensor,
        actions: Tensor,
        incoming: Tensor,
    ) -> Tensor:
        """Supervise the action and observation predictors with Equations 12--13."""
        own_action = F.one_hot(
            actions[:, self.agent_index].long(), self.action_dim,
        ).to(dtype=obs.dtype)
        _, other_logits, predicted_next = self.imagine(obs, incoming, own_action)
        predicted_actions = torch.softmax(other_logits, dim=-1)
        target_actions = F.one_hot(
            self._other_actions(actions).long(), self.action_dim,
        ).to(dtype=obs.dtype)
        action_loss = F.mse_loss(predicted_actions, target_actions)
        observation_loss = F.mse_loss(predicted_next - obs, next_obs - obs)
        return action_loss + observation_loss

    def action(
        self,
        obs: Tensor,
        incoming: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample the actor without constructing an outgoing message."""
        logits = self.policy_head(torch.cat([obs, self._context(incoming)], dim=-1))
        return gumbel_policy_sample(
            logits,
            action_dim=self.action_dim,
            hard=True,
            deterministic=deterministic,
        )

    def sample(
        self,
        obs: Tensor,
        incoming: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample one action, then build the message from that executed action."""
        one_hot, action_index, logits = self.action(
            obs, incoming, deterministic=deterministic,
        )
        outgoing = self.messages(obs, incoming, one_hot)
        return one_hot, action_index, logits, outgoing


class _IntentionSharingAgent(nn.Module):
    """One independently parameterized actor-critic pair and its target copies."""

    def __init__(
        self,
        agent_index: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int,
        hidden_dim: int,
        imagination_horizon: int,
    ) -> None:
        super().__init__()
        self.policy = IntentionSharingPolicy(
            agent_index,
            n_agents,
            obs_dim,
            action_dim,
            message_dim,
            hidden_dim,
            imagination_horizon,
        )
        self.critic = CentralizedMLPCritic(n_agents, obs_dim, action_dim, hidden_dim)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_policy.requires_grad_(False)
        self.target_critic.requires_grad_(False)

    def soft_update(self, tau: float) -> None:
        soft_update_module(self.target_policy, self.policy, tau)
        soft_update_module(self.target_critic, self.critic, tau)


class IntentionSharingLearner(nn.Module):
    """Independent MADDPG actors/critics with differentiable intention channels."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int = 3,
        hidden_dim: int = 128,
        imagination_horizon: int = 5,
        config: IntentionSharingConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.config = config or IntentionSharingConfig()
        self.agents = nn.ModuleList(
            [
                _IntentionSharingAgent(
                    index,
                    n_agents,
                    obs_dim,
                    action_dim,
                    message_dim,
                    hidden_dim,
                    imagination_horizon,
                )
                for index in range(n_agents)
            ]
        )
        self.policy_optimizers = [
            torch.optim.Adam(agent.policy.parameters(), lr=self.config.learning_rate)
            for agent in self.agents
        ]
        self.critic_optimizers = [
            torch.optim.Adam(agent.critic.parameters(), lr=self.config.learning_rate)
            for agent in self.agents
        ]
        self.replay = _ReplayBuffer(
            self.config.replay_capacity, n_agents, obs_dim, message_dim,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def __repr__(self) -> str:
        return (
            f"IntentionSharingLearner(n_agents={self.n_agents}, obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, message_dim={self.message_dim})"
        )

    def sample(
        self,
        obs: Tensor,
        incoming: Tensor,
        *,
        target: bool = False,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample all independent policies while retaining joint tensor ordering."""
        outputs = []
        for index, agent in enumerate(self.agents):
            policy = agent.target_policy if target else agent.policy
            outputs.append(policy.sample(obs[:, index], incoming, deterministic=deterministic))
        return tuple(torch.stack(items, dim=1) for items in zip(*outputs, strict=True))

    def _sample_actions(
        self,
        obs: Tensor,
        incoming: Tensor,
        *,
        target: bool = False,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        outputs = []
        for index, agent in enumerate(self.agents):
            policy = agent.target_policy if target else agent.policy
            outputs.append(policy.action(obs[:, index], incoming, deterministic=deterministic))
        return tuple(torch.stack(items, dim=1) for items in zip(*outputs, strict=True))

    def messages_for_actions(
        self,
        obs: Tensor,
        incoming: Tensor,
        actions: Tensor,
        *,
        detach_trajectory: bool = False,
    ) -> Tensor:
        """Build outgoing messages from externally selected executed one-hot actions."""
        return torch.stack(
            [
                agent.policy.messages(
                    obs[:, index],
                    incoming,
                    actions[:, index],
                    detach_trajectory=detach_trajectory,
                )
                for index, agent in enumerate(self.agents)
            ],
            dim=1,
        )

    def store(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_obs: np.ndarray,
        dones: np.ndarray | bool,
        messages_seen: np.ndarray,
        messages_written: np.ndarray,
    ) -> None:
        self.replay.add(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            messages_seen,
            messages_written,
        )

    def ready(self) -> bool:
        return len(self.replay) >= max(
            self.config.batch_size, self.config.minimum_replay_size,
        )

    @staticmethod
    def _action_with_detached_parameters(
        policy: IntentionSharingPolicy,
        obs: Tensor,
        incoming: Tensor,
    ) -> Tensor:
        """Differentiate a receiver action only with respect to its incoming messages."""
        actor_input = torch.cat([obs, policy._context(incoming)], dim=-1)
        detached_parameters = {
            name: parameter.detach()
            for name, parameter in policy.policy_head.named_parameters()
        }
        logits = functional_call(policy.policy_head, detached_parameters, (actor_input,))
        return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

    def update(self, batch: _ReplayBatch | None = None) -> IntentionSharingUpdate:
        """Apply Equations 9--13 with independent actor and critic ownership."""
        if batch is None:
            if not self.ready():
                raise RuntimeError("replay does not contain enough transitions")
            batch = self.replay.sample(self.config.batch_size, self.device)
        replay_actions = F.one_hot(batch.actions.long(), self.action_dim).to(batch.obs.dtype)

        with torch.no_grad():
            next_target_actions, _, _ = self._sample_actions(
                batch.next_obs,
                batch.messages_written,
                target=True,
                deterministic=True,
            )
        critic_losses = []
        for index, (agent, optimizer) in enumerate(
            zip(self.agents, self.critic_optimizers, strict=True),
        ):
            with torch.no_grad():
                target = batch.rewards[:, index] + self.config.gamma * (
                    1.0 - batch.dones[:, index]
                ) * agent.target_critic(batch.next_obs, next_target_actions)
            critic_loss = F.mse_loss(agent.critic(batch.obs, replay_actions), target)
            optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), self.config.gradient_clip)
            optimizer.step()
            critic_losses.append(critic_loss.detach())

        for agent in self.agents:
            agent.critic.requires_grad_(False)
        policy_actions, _, logits = self._sample_actions(batch.obs, batch.messages_seen)
        written = self.messages_for_actions(
            batch.obs,
            batch.messages_seen,
            replay_actions,
            detach_trajectory=True,
        )
        current_policy_loss = batch.obs.new_zeros(())
        message_policy_loss = batch.obs.new_zeros(())
        model_loss = batch.obs.new_zeros(())
        for index, agent in enumerate(self.agents):
            counterfactual_actions = torch.stack(
                [
                    policy_actions[:, other]
                    if other == index
                    else replay_actions[:, other].detach()
                    for other in range(self.n_agents)
                ],
                dim=1,
            )
            current_policy_loss = current_policy_loss - agent.critic(
                batch.obs, counterfactual_actions,
            ).mean()
            receiver_action = self._action_with_detached_parameters(
                agent.policy, batch.next_obs[:, index], written,
            )
            receiver_counterfactual = torch.stack(
                [
                    receiver_action
                    if other == index
                    else next_target_actions[:, other].detach()
                    for other in range(self.n_agents)
                ],
                dim=1,
            )
            message_policy_loss = message_policy_loss - (
                (1.0 - batch.dones[:, index])
                * agent.critic(batch.next_obs, receiver_counterfactual)
            ).mean()
            model_loss = model_loss + agent.policy.model_loss(
                batch.obs[:, index],
                batch.next_obs[:, index],
                batch.actions,
                batch.messages_seen,
            )
        message_policy_loss = message_policy_loss / self.n_agents
        regularization = logits.square().mean()
        policy_loss = (
            current_policy_loss
            + message_policy_loss
            + self.config.policy_regularization * regularization
            + self.config.model_loss_weight * model_loss
        )
        for optimizer in self.policy_optimizers:
            optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        for agent, optimizer in zip(self.agents, self.policy_optimizers, strict=True):
            nn.utils.clip_grad_norm_(agent.policy.parameters(), self.config.gradient_clip)
            optimizer.step()
            agent.critic.requires_grad_(True)
            agent.soft_update(self.config.tau)

        return IntentionSharingUpdate(
            critic_loss=float(torch.stack(critic_losses).mean()),
            policy_loss=float(policy_loss.detach()),
            model_loss=float(model_loss.detach()),
        )


__all__ = [
    "IntentionSharingConfig",
    "IntentionSharingLearner",
    "IntentionSharingPolicy",
    "IntentionSharingUpdate",
]
