"""Deep Independent Q-Learning as used by Tampuu et al. (2017).

Paper: Tampuu et al., "Multiagent Cooperation and Competition with Deep
Reinforcement Learning," PLOS ONE 12(4), 2017. Historical source:
``NeuroCSUT/DeepMind-Atari-Deep-Q-Learner-2Player@feb0b8a``.

Model: every agent owns a separate online DQN, target DQN, centered-RMSProp
optimizer, and replay stream. Each independently minimizes the paper's one-step
target ``r_i + gamma max_a Q_i^-(o_i', a)`` and treats the other learners as part
of its environment. Hard target copies, clipped rewards, update cadence, and
step-based epsilon annealing follow the release. There is no parameter sharing,
joint value, centralized input, or cross-agent gradient.

Adaptation: the paper's Pong experiment uses four stacked 84x84 frames and an
Atari convolutional frontend. modMARL environments expose vector observations,
so this implementation uses the release's own two-layer 128-unit MLP fallback;
the independent DQN algorithm and optimization are unchanged. Cooperative tasks
provide the same team reward to each otherwise independent learner.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from ..common.nn import build_mlp
from ..common.replay import ReplayBatch


@dataclass(frozen=True)
class IQLUpdate:
    """Diagnostics from one independent learner update."""

    loss: float
    mean_q: float
    mean_target: float
    mean_absolute_td_error: float


class IndependentQNetwork(nn.Module):
    """The release's vector-input DQN: two 128-unit ReLU layers."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, [hidden_dim, hidden_dim], action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


class _ReleasedCenteredRMSProp(Optimizer):
    """Centered RMSProp equation used by the released Torch7 DQN.

    Unlike ``torch.optim.RMSprop``, the release adds epsilon inside the square
    root: ``sqrt(E[g^2] - E[g]^2 + eps)``.
    """

    def __init__(self, parameters, *, lr: float, alpha: float = 0.95, eps: float = 0.01) -> None:
        super().__init__(parameters, {"lr": lr, "alpha": alpha, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("released centered RMSProp does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["mean_gradient"] = torch.zeros_like(parameter)
                    state["mean_square"] = torch.zeros_like(parameter)
                mean_gradient = state["mean_gradient"]
                mean_square = state["mean_square"]
                alpha = group["alpha"]
                mean_gradient.mul_(alpha).add_(gradient, alpha=1.0 - alpha)
                mean_square.mul_(alpha).addcmul_(gradient, gradient, value=1.0 - alpha)
                denominator = (mean_square - mean_gradient.square() + group["eps"]).sqrt()
                parameter.addcdiv_(gradient, denominator, value=-group["lr"])
        return loss


class IQLAgent(nn.Module):
    """Complete collection of autonomous DQN learners, one per environment agent."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        target_update_interval: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_anneal_steps: int = 1_000_000,
        learn_start: int = 50_000,
        reward_clip: float = 1.0,
    ) -> None:
        super().__init__()
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        if target_update_interval < 1 or epsilon_anneal_steps < 1:
            raise ValueError("target and epsilon intervals must be positive")
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.learn_start = learn_start
        self.reward_clip = reward_clip
        self.q_networks = nn.ModuleList(
            IndependentQNetwork(obs_dim, action_dim, hidden_dim) for _ in range(n_agents)
        )
        self.target_q_networks = copy.deepcopy(self.q_networks)
        self.target_q_networks.requires_grad_(False)
        self.optimizers = [
            _ReleasedCenteredRMSProp(network.parameters(), lr=learning_rate)
            for network in self.q_networks
        ]

    def epsilon(self, env_step: int) -> float:
        """Paper schedule: anneal from 1.0 to 0.05 after replay warm-up."""
        elapsed = max(0, env_step - self.learn_start)
        fraction = min(elapsed / self.epsilon_anneal_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def act(self, obs: Tensor, epsilon: float) -> Tensor:
        """Select independent epsilon-greedy actions from ``obs (n_agents, obs_dim)``."""
        if obs.shape[0] != self.n_agents:
            raise ValueError(f"expected {self.n_agents} observations, got {obs.shape[0]}")
        actions = []
        for index, network in enumerate(self.q_networks):
            if torch.rand((), device=obs.device) < epsilon:
                action = torch.randint(self.action_dim, (), device=obs.device)
            else:
                q_values = network(obs[index])
                best = torch.nonzero(q_values == q_values.max(), as_tuple=False).flatten()
                action = best[torch.randint(len(best), (), device=obs.device)]
            actions.append(action)
        return torch.stack(actions)

    def update_agent(self, agent_index: int, batch: ReplayBatch) -> IQLUpdate:
        """Apply one independent DQN update from one agent's private replay batch.

        ``batch`` is a generic replay batch with singleton agent axis:
        observations ``(B, 1, obs_dim)`` and actions ``(B, 1)``. The returned
        diagnostics include the summed Huber loss whose unit threshold reproduces
        the release's clipped TD-error gradient.
        """
        network = self.q_networks[agent_index]
        target_network = self.target_q_networks[agent_index]
        obs = batch.obs[:, 0]
        next_obs = batch.next_obs[:, 0]
        actions = batch.actions[:, 0].long()
        rewards = batch.rewards.clamp(-self.reward_clip, self.reward_clip)

        chosen_q = network(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = target_network(next_obs).max(dim=-1).values
            target = rewards + self.gamma * (1.0 - batch.dones) * next_q
        # Torch7 receives the clipped TD errors as gradOutput and therefore sums
        # them across the minibatch; reduction="sum" preserves that update.
        loss = F.smooth_l1_loss(chosen_q, target, reduction="sum")

        optimizer = self.optimizers[agent_index]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return IQLUpdate(
            loss=float(loss.detach()),
            mean_q=float(chosen_q.detach().mean()),
            mean_target=float(target.detach().mean()),
            mean_absolute_td_error=float((chosen_q.detach() - target.detach()).abs().mean()),
        )

    def maybe_update_targets(self, env_step: int) -> bool:
        """Hard-copy every target at the release's environment-step cadence."""
        # The release increments numSteps before checking ``numSteps % target_q == 1``.
        if env_step <= 0 or (env_step - 1) % self.target_update_interval != 0:
            return False
        for target, online in zip(self.target_q_networks, self.q_networks):
            target.load_state_dict(online.state_dict())
        return True

    def update_targets(self) -> None:
        """Force a hard target copy, useful for checkpoint restoration and tests."""
        for target, online in zip(self.target_q_networks, self.q_networks):
            target.load_state_dict(online.state_dict())


__all__ = ["IQLAgent", "IQLUpdate", "IndependentQNetwork"]
