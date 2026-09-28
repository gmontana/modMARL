"""MADDPG-M implementation.

Original paper:
Ozsel Kilinc and Giovanni Montana. "Multi-agent Deep Reinforcement Learning with Extremely
Noisy Observations." NeurIPS 2018 Deep Reinforcement Learning Workshop. arXiv:1812.00922.

MADDPG-M augments MADDPG with a learned communication medium and a two-level policy. The
top-level communication policy ν_i emits a scalar "willingness" to broadcast; an argmax over
the agents selects whose observation is placed in the shared medium. The bottom-level action
policy μ_i(o_i, m) emits the paper's four continuous direction magnitudes. The two levels are trained
with different rewards: μ from an intrinsic reward (reaching the landmarks encoded in the
medium) and ν from the extrinsic task reward — so ν learns to broadcast the informative
(gifted) agent's observation.

This is the paper's broadcasting variant. Communication is trained centrally at a slower
time scale, while action policies and critics are trained independently from intrinsic
rewards. In particular, Equation 9 deliberately uses the local critic Q(o_i, m_i, a_i): only
the communication critic is centralised. Policies use two 64-unit layers and critics use two
128-unit layers, matching the experimental architecture.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ..common.nn import build_mlp
from ..components import soft_update_module


class CommunicationPolicy(nn.Module):
    """ν: maps an agent's observation to a scalar broadcast willingness in [0, 1]."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, [hidden_dim, hidden_dim], 1)

    def forward(self, obs: Tensor) -> Tensor:
        return torch.sigmoid(self.net(obs))


class ActionPolicy(nn.Module):
    """μ: maps (observation, medium) to continuous direction magnitudes in [0, 1]."""

    def __init__(self, obs_dim: int, medium_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + medium_dim, [hidden_dim, hidden_dim], action_dim)

    def forward(self, obs: Tensor, medium: Tensor) -> Tensor:
        return torch.sigmoid(self.net(torch.cat([obs, medium], dim=-1)))


class CommCritic(nn.Module):
    """Q^ν: centralized value of the joint (observations, broadcast willingnesses)."""

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = build_mlp(n_agents * obs_dim + n_agents, [hidden_dim, hidden_dim], 1)

    def forward(self, obs_all: Tensor, comm_all: Tensor) -> Tensor:
        # obs_all: (batch, n_agents, obs_dim), comm_all: (batch, n_agents)
        flat_obs = obs_all.reshape(obs_all.shape[0], -1)
        return self.net(torch.cat([flat_obs, comm_all], dim=-1)).squeeze(-1)


class ActionCritic(nn.Module):
    """Q^μ: decentralized value of (observation, medium, action) for one agent."""

    def __init__(self, obs_dim: int, medium_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + medium_dim + action_dim, [hidden_dim, hidden_dim], 1)

    def forward(self, obs: Tensor, medium: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([obs, medium, action], dim=-1)).squeeze(-1)


class MADDPGMAgent(nn.Module):
    """One MADDPG-M agent: a communication policy and an action policy, each with its own critic."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.comm_policy = CommunicationPolicy(obs_dim, hidden_dim)
        self.action_policy = ActionPolicy(obs_dim, obs_dim, action_dim, hidden_dim)
        self.comm_critic = CommCritic(n_agents, obs_dim, critic_hidden_dim)
        self.action_critic = ActionCritic(obs_dim, obs_dim, action_dim, critic_hidden_dim)
        self.target_comm_policy = copy.deepcopy(self.comm_policy)
        self.target_action_policy = copy.deepcopy(self.action_policy)
        self.target_comm_critic = copy.deepcopy(self.comm_critic)
        self.target_action_critic = copy.deepcopy(self.action_critic)

    def soft_update(self, tau: float) -> None:
        soft_update_module(self.target_comm_policy, self.comm_policy, tau)
        soft_update_module(self.target_action_policy, self.action_policy, tau)
        soft_update_module(self.target_comm_critic, self.comm_critic, tau)
        soft_update_module(self.target_action_critic, self.action_critic, tau)


__all__ = ["ActionCritic", "ActionPolicy", "CommCritic", "CommunicationPolicy", "MADDPGMAgent"]
