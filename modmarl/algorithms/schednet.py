"""SchedNet's bandwidth-constrained actor and centralized two-value critic.

Original paper:
Daewoo Kim, Sangwoo Moon, David Hostallero, Wan Ju Kang, Taeyoung Lee, Kyunghwan Son,
Yung Yi. "Learning to Schedule Communication in Multi-agent Reinforcement Learning."
International Conference on Learning Representations (ICLR), 2019. arXiv:1902.01554.

Model: every agent owns an independent weight generator, encoder, and categorical action
selector. WSA routes exactly the top-k encoded messages to all agents. One centralized critic
shares its first two state layers between V(s) and Q(s,w), then uses distinct third layers and
heads. Target ownership is limited to the weight generator and critic.

Invariants: execution uses only local observations plus the scheduled broadcast; schedules have
exactly k speakers; action-policy gradients do not enter the weight generators; target modules
never receive gradients. Tensor order is always (batch, agent, feature).

Interface: ``SchedNetAgent.act`` performs the distributed protocol, while ``soft_update`` is the
only target mutation. The trainer owns replay, paper Equation (4), deterministic policy gradients,
and exploration.

Official code: https://github.com/rhoowd/sched_net, reference commit
ffa03007cc654000a859856401231a986a01fbd0 (cited in the paper itself).

Source reconciliation: Appendix B.2 says one action-selector hidden layer and three encoder/WG
layers, but the pinned executable release uses three, one, and two respectively; this module uses
that released topology. The release also flattens all per-agent probabilities and logs their sum;
paper Equation (4) instead requires the log of the factorized joint policy, implemented by summing
per-agent log-probabilities. Finally, the release bootstraps Q(s,w) across terminal states; the
trainer applies the mathematically required terminal mask to both critic targets.

The action selector is a softmax policy trained by paper Equation (4), the stochastic policy
gradient weighted by that critic's TD error, with the release's 0.01 entropy bonus:
``-(log pi(u|o,c) * td_error + 0.01 * entropy)``. The weight generator is deterministic and
trained through the schedule head. The release evaluates ``grad_w Q(s,w)`` at
the replayed priorities (``agent.py:update_ac``), then applies that detached
gradient to the current weight generator. This differs from the usual DDPG
evaluation at ``w = mu(o)`` suggested by paper Section 3.3.1. The trainer follows
the released evaluation point, averaging the batch loss as for its other losses.

Historical curves use three-agent navigation; the frozen learning recipe uses two-agent
one-step signaling. These bounded checks do not reproduce the paper's predator-prey benchmark.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ..components import soft_update_module


def _released_linear(
    input_dim: int,
    output_dim: int,
    *,
    bias: bool = True,
) -> nn.Linear:
    """Build the release's Normal(0, 0.1) affine layer with 0.1 bias."""
    layer = nn.Linear(input_dim, output_dim, bias=bias)
    nn.init.normal_(layer.weight, mean=0.0, std=0.1)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.1)
    return layer


def _weight_output(input_dim: int) -> nn.Linear:
    """Build the WG output layer, whose TensorFlow release uses Glorot and zero bias."""
    layer = nn.Linear(input_dim, 1)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def top_k_schedule(weights: Tensor, k: int) -> Tensor:
    """Weight-based scheduler (WSA, Top(k)): a 0/1 mask keeping the k highest-weight agents.

    weights: (..., n_agents) scheduling weights. Returns a 0/1 tensor of the same shape with
    exactly min(k, n_agents) ones along the last axis.
    """
    k = min(k, weights.shape[-1])
    indices = torch.topk(weights, k, dim=-1).indices
    mask = torch.zeros_like(weights)
    mask.scatter_(-1, indices, 1.0)
    return mask


def aggregate_broadcast(messages: Tensor, schedule: Tensor) -> Tensor:
    """Compact scheduled messages in sender order and broadcast them to every agent.

    messages: (batch, n_agents, message_dim); schedule: (batch, n_agents) 0/1.
    Every batch row must schedule the same k senders. Returns
    ``(batch, n_agents, k * message_dim)``.
    """
    batch, n_agents, message_dim = messages.shape
    counts = schedule.sum(dim=-1).long()
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("every batch row must schedule the same number of senders")
    k = int(counts[0].item())
    selected = messages[schedule.bool()].view(batch, k, message_dim)
    channel = selected.reshape(batch, 1, k * message_dim)
    return channel.expand(-1, n_agents, -1)


class WeightGenerator(nn.Module):
    """Independent released f_wg blocks: ``(B, N, O) -> (B, N)``."""

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    _released_linear(obs_dim, hidden_dim),
                    nn.ReLU(),
                    _released_linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    _weight_output(hidden_dim),
                    nn.Sigmoid(),
                )
                for _ in range(n_agents)
            ]
        )

    def forward(self, obs: Tensor) -> Tensor:
        if obs.ndim != 3 or obs.shape[1] != self.n_agents:
            raise ValueError(f"obs must have shape (batch, {self.n_agents}, obs_dim)")
        return torch.stack(
            [network(obs[:, index]).squeeze(-1) for index, network in enumerate(self.networks)],
            dim=1,
        )


class MessageEncoder(nn.Module):
    """Independent released f_enc blocks: ``(B, N, O) -> (B, N, M)``."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        message_dim: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    _released_linear(obs_dim, hidden_dim),
                    nn.ReLU(),
                    _released_linear(hidden_dim, message_dim),
                    nn.ReLU(),
                )
                for _ in range(n_agents)
            ]
        )

    def forward(self, obs: Tensor) -> Tensor:
        if obs.ndim != 3 or obs.shape[1] != self.n_agents:
            raise ValueError(f"obs must have shape (batch, {self.n_agents}, obs_dim)")
        return torch.stack(
            [network(obs[:, index]) for index, network in enumerate(self.networks)],
            dim=1,
        )


class ActionSelector(nn.Module):
    """Independent released f_as blocks over local observation and broadcast.

    The released selector is a softmax head sampled with ``np.random.choice`` and trained by
    the stochastic policy gradient of paper Equation (4) -- not a deterministic head with a
    straight-through estimator.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        channel_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        input_dim = obs_dim + channel_dim
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    _released_linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    _released_linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    _released_linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    _released_linear(hidden_dim, action_dim),
                )
                for _ in range(n_agents)
            ]
        )

    def forward(self, obs: Tensor, message: Tensor) -> Tensor:
        if obs.ndim != 3 or obs.shape[1] != self.n_agents:
            raise ValueError(f"obs must have shape (batch, {self.n_agents}, obs_dim)")
        inputs = torch.cat([obs, message], dim=-1)
        return torch.stack(
            [network(inputs[:, index]) for index, network in enumerate(self.networks)],
            dim=1,
        )

    def distribution(self, obs: Tensor, message: Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(obs, message))

    def sample(
        self, obs: Tensor, message: Tensor, *, deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(action, log_prob)``, each ``(batch, n_agents)``."""
        distribution = self.distribution(obs, message)
        action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action)


class SchedNetCritic(nn.Module):
    """Released critic: one trunk over the state with a value head and a schedule head.

    Paper Section 3.3.1: "To share common features between V(s) and Q(s,w) and perform
    efficient training, we use shared parameters in the lower layers of the neural network
    between the two functions." The released ``generate_critic_network`` builds both heads
    off the same first two hidden layers and trains them with the single loss
    ``mean(td_errors ** 2 + sch_td_errors ** 2)``. Neither head takes actions --
    ``use_action_in_critic`` is False in every published run.
    """

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared_1 = _released_linear(n_agents * obs_dim, hidden_dim)
        self.shared_2 = _released_linear(hidden_dim, hidden_dim)
        self.value_hidden = _released_linear(hidden_dim, hidden_dim)
        self.schedule_hidden = _released_linear(hidden_dim + n_agents, hidden_dim)
        self.value_head = _released_linear(hidden_dim, 1, bias=False)
        self.schedule_head = _released_linear(hidden_dim, 1, bias=False)

    def forward(self, obs_all: Tensor, weights_all: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(V(s), Q_sched(s, w))``, each ``(batch,)``."""
        hidden_1 = torch.relu(self.shared_1(obs_all.reshape(obs_all.shape[0], -1)))
        hidden_2 = torch.relu(self.shared_2(hidden_1))
        value = self.value_head(torch.relu(self.value_hidden(hidden_2))).squeeze(-1)
        schedule_hidden = torch.relu(
            self.schedule_hidden(torch.cat([hidden_2, weights_all], dim=-1))
        )
        schedule_value = self.schedule_head(schedule_hidden).squeeze(-1)
        return value, schedule_value


class SchedNetAgent(nn.Module):
    """Independent actor blocks and the released centralized two-head critic."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int = 2,
        actor_hidden_dim: int = 32,
        critic_hidden_dim: int = 64,
        scheduler_hidden_dim: int = 32,
        bandwidth: int | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.bandwidth = max(1, n_agents // 2) if bandwidth is None else min(bandwidth, n_agents)
        self.weight_generator = WeightGenerator(n_agents, obs_dim, scheduler_hidden_dim)
        self.message_encoder = MessageEncoder(n_agents, obs_dim, message_dim, actor_hidden_dim)
        self.action_selector = ActionSelector(
            n_agents,
            obs_dim,
            self.bandwidth * message_dim,
            action_dim,
            actor_hidden_dim,
        )
        self.critic = SchedNetCritic(n_agents, obs_dim, critic_hidden_dim)

        self.target_weight_generator = copy.deepcopy(self.weight_generator)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_weight_generator.requires_grad_(False)
        self.target_critic.requires_grad_(False)

    def act(
        self,
        obs: Tensor,
        k: int,
        *,
        priorities: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full pipeline for one step. obs: (batch, n_agents, obs_dim).

        Returns (action, log_prob, weights, schedule):
        weights: (batch, n_agents) the scheduling weights (with optional exploration noise);
        schedule: (batch, n_agents) the 0/1 top-k schedule the messages were aggregated under.
        """
        if min(k, obs.shape[1]) != self.bandwidth:
            raise ValueError(f"this SchedNet was built for bandwidth={self.bandwidth}, got k={k}")
        weights = self.weight_generator(obs) if priorities is None else priorities
        if weights.shape != obs.shape[:2]:
            raise ValueError("priorities must have shape (batch, n_agents)")
        schedule = top_k_schedule(weights, k)
        messages = self.message_encoder(obs)
        broadcast = aggregate_broadcast(messages, schedule)
        action, log_prob = self.action_selector.sample(
            obs, broadcast, deterministic=deterministic,
        )
        return action, log_prob, weights, schedule

    def broadcast_for(self, obs: Tensor, schedule: Tensor) -> Tensor:
        """Re-encode and aggregate the messages the given schedule would deliver."""
        return aggregate_broadcast(self.message_encoder(obs), schedule)

    def soft_update(self, tau: float) -> None:
        """The release soft-updates only the slow target critic and weight generator."""
        soft_update_module(self.target_weight_generator, self.weight_generator, tau)
        soft_update_module(self.target_critic, self.critic, tau)


__all__ = [
    "ActionSelector",
    "MessageEncoder",
    "SchedNetAgent",
    "SchedNetCritic",
    "WeightGenerator",
    "aggregate_broadcast",
    "top_k_schedule",
]
