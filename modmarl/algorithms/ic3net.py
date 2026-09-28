"""IC3Net with individualized rewards and the released REINFORCE learner.

Original paper:
Amanpreet Singh, Tushar Jain, Sainbayar Sukhbaatar. "Learning when to Communicate at
Scale in Multiagent Cooperative and Competitive Tasks." International Conference on
Learning Representations (ICLR), 2019. arXiv:1812.09755.
Official code: https://github.com/IC3Net/IC3Net (MIT), reference revision
69b7e0ce51a79def593abfef1a976f43e5e13f75.

A shared LSTM policy where each agent, at every step, also takes a discrete binary
"talk / stay silent" action from a second policy head. The communication vector fed
into agent i's LSTM update is C applied to the mean of the *other* agents' hidden
states, each multiplied by the gate that agent sampled on the previous step:

    g_j^{t+1} = f_g(h_j^t)                                  (gate decided one step ahead)
    c_i       = C * (1/(J-1)) * sum_{j != i} g_j * h_j      (sender-side gating)
    h', s'    = LSTM(e(o) + c, (h, s))                      (weights shared across agents)

Action, gate, and value heads all read the updated hidden state. Gates start silent
at each episode's first step, as in the reference trainer.

The complete action is the environment action plus the binary communication action.
The released objective uses each agent's own discounted return (``mean_ratio=0``),
the joint log-probability of both choices, a learned local value baseline, RMSProp
at 1e-3 (alpha 0.97, epsilon 1e-6), gamma 1, value coefficient 0.01, 128-unit
paper-task recurrence, and recurrent detachment every 10 steps. Rollouts therefore
carry explicit reward, active-agent, and individual-continuation vectors; a
cooperative environment may provide identical rewards, but the learner never
collapses them to a team scalar. The bundled validation environments currently
emit shared rewards, so the learning curves exercise the cooperative IC3Net
setting; the per-agent API and golden tests retain the paper's individualized-
reward objective.

Paper/code reconciliation: sender-only gating follows the paper's equation. The
release additionally prevents a silent sender from receiving because it reuses its
alive-agent mask; that unrelated coupling is not retained. The released biased C
map and all other underspecified architecture details are retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class IC3NetCell(nn.Module):
    """One IC3Net step: gated-mean communication + shared LSTM cell + the three heads.

    forward(obs (B, n, obs_dim), hidden (B, n, H), cell (B, n, H), gate_prev (B, n))
      -> (action_logits (B, n, A), gate_logits (B, n, 2), value (B, n),
          hidden' (B, n, H), cell' (B, n, H))
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.comm = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.gate_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: Tensor, hidden: Tensor, cell: Tensor, gate_prev: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, n_agents, _ = obs.shape
        gated = hidden * gate_prev.to(hidden.dtype).unsqueeze(-1)
        if n_agents == 1:
            comm_input = torch.zeros_like(hidden)
        else:
            comm_input = (gated.sum(dim=1, keepdim=True) - gated) / (n_agents - 1)
        x = self.encoder(obs) + self.comm(comm_input)

        flat = lambda t: t.reshape(batch * n_agents, self.hidden_dim)  # noqa: E731
        new_hidden, new_cell = self.lstm(
            x.reshape(batch * n_agents, self.hidden_dim), (flat(hidden), flat(cell)),
        )
        new_hidden = new_hidden.view(batch, n_agents, self.hidden_dim)
        new_cell = new_cell.view(batch, n_agents, self.hidden_dim)

        action_logits = self.policy_head(new_hidden)
        gate_logits = self.gate_head(new_hidden)
        value = self.value_head(new_hidden).squeeze(-1)
        return action_logits, gate_logits, value, new_hidden, new_cell


class IC3NetAgent(nn.Module):
    """IC3Net recurrent policy, value baseline, and RMSProp learner."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        learning_rate: float = 1e-3,
        gamma: float = 1.0,
        value_coefficient: float = 0.01,
        entropy_coefficient: float = 0.0,
        detach_gap: int = 10,
    ) -> None:
        super().__init__()
        self.cell = IC3NetCell(obs_dim, action_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.detach_gap = detach_gap
        self.optimizer = torch.optim.RMSprop(
            self.parameters(), lr=learning_rate, alpha=0.97, eps=1e-6,
        )

    def init_state(self, batch: int, n_agents: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """Zero hidden/cell states and all-silent gates for an episode start."""
        hidden = torch.zeros(batch, n_agents, self.hidden_dim, device=device)
        cell = torch.zeros(batch, n_agents, self.hidden_dim, device=device)
        gate_prev = torch.zeros(batch, n_agents, dtype=torch.long, device=device)
        return hidden, cell, gate_prev

    def act(
        self,
        obs: Tensor,
        hidden: Tensor,
        cell: Tensor,
        gate_prev: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample an env action and next step's gate; returns log-probs and value too.

        -> (action (B, n), log_prob (B, n), gate (B, n), gate_log_prob (B, n),
            value (B, n), hidden' , cell')
        """
        action_logits, gate_logits, value, new_hidden, new_cell = self.cell(obs, hidden, cell, gate_prev)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        action = action_logits.argmax(dim=-1) if deterministic else action_dist.sample()
        gate = gate_logits.argmax(dim=-1) if deterministic else gate_dist.sample()
        return action, action_dist.log_prob(action), gate, gate_dist.log_prob(gate), value, new_hidden, new_cell

    def evaluate_step(
        self, obs: Tensor, hidden: Tensor, cell: Tensor, gate_prev: Tensor, action: Tensor, gate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Re-score a stored on-policy rollout step under the current parameters.

        -> (joint_log_prob (B, n) = log pi(a) + log pi(g), entropy (B, n),
            value (B, n), hidden', cell')
        """
        action_logits, gate_logits, value, new_hidden, new_cell = self.cell(obs, hidden, cell, gate_prev)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        joint_log_prob = action_dist.log_prob(action) + gate_dist.log_prob(gate)
        entropy = action_dist.entropy() + gate_dist.entropy()
        return joint_log_prob, entropy, value, new_hidden, new_cell

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        gates: Tensor,
        rewards: Tensor,
        mask: Tensor,
        *,
        alive: Tensor | None = None,
        continuation: Tensor | None = None,
    ) -> IC3NetUpdate:
        """Apply the released individualized REINFORCE update.

        ``obs`` is ``(batch, time, agents, obs_dim)`` and every other tensor is
        ``(batch, time, agents)``. Rewards must remain agent-specific at the API.
        """
        if rewards.shape != actions.shape or mask.shape != actions.shape:
            raise ValueError("IC3Net requires per-agent rewards and masks")
        if alive is None:
            alive = mask
        if continuation is None:
            continuation = mask
        batch_size, horizon, n_agents = actions.shape
        returns = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[:, 0])
        for step in range(horizon - 1, -1, -1):
            running = (
                rewards[:, step]
                + self.gamma * running * continuation[:, step]
            ) * mask[:, step]
            returns[:, step] = running

        hidden, cell, gate_previous = self.init_state(batch_size, n_agents, obs.device)
        log_probs, entropies, values = [], [], []
        for step in range(horizon):
            log_prob, entropy, value, hidden, cell = self.evaluate_step(
                obs[:, step],
                hidden,
                cell,
                gate_previous,
                actions[:, step],
                gates[:, step],
            )
            gate_previous = gates[:, step]
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            if (step + 1) % self.detach_gap == 0:
                hidden, cell = hidden.detach(), cell.detach()
        log_probs_tensor = torch.stack(log_probs, dim=1)
        entropy_tensor = torch.stack(entropies, dim=1)
        values_tensor = torch.stack(values, dim=1)

        advantage = returns - values_tensor.detach()
        loss_mask = mask * alive
        transitions = mask.any(dim=-1).sum().clamp_min(1).to(obs.dtype)
        action_loss = -(log_probs_tensor * advantage * loss_mask).sum() / transitions
        value_loss = ((values_tensor - returns).square() * loss_mask).sum() / transitions
        entropy = (entropy_tensor * loss_mask).sum() / transitions
        loss = action_loss + self.value_coefficient * value_loss
        loss = loss - self.entropy_coefficient * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return IC3NetUpdate(
            action_loss=float(action_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropy.detach()),
            return_mean=float(returns[loss_mask.bool()].mean()),
        )


@dataclass(frozen=True)
class IC3NetUpdate:
    """Diagnostics from one released-style rollout update."""

    action_loss: float
    value_loss: float
    entropy: float
    return_mean: float


__all__ = ["IC3NetAgent", "IC3NetCell", "IC3NetUpdate"]
