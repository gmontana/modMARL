"""DIAL and RIAL from Foerster et al., "Learning to Communicate with Deep
Multi-Agent Reinforcement Learning" (NeurIPS 2016; arXiv:1605.06676v2).

Model: every agent runs a recurrent C-Net (Section 6.1). Its input embedding is
the sum z = TaskEncoder(o) + MessageMLP(m_{t-1}) + Lookup(u_{t-1}) + Lookup(a),
followed by a two-layer 128-unit GRU and a two-layer MLP that outputs
|U| environment Q-values plus a message head.

- **DIAL** (Section 5.2, Algorithm 1): the message head is a real-valued
  vector passed through the discretise/regularise unit (DRU). In centralized
  training the receiver sees Logistic(m + N(0, sigma^2)), so the TD loss of the
  receiver backpropagates into the sender across agents and time. In
  decentralized execution the channel is binary.
- **RIAL** (Section 5.1): the message head holds Q-values over 2^bits discrete
  messages, chosen epsilon-greedily and trained with its own DQN loss; no
  gradient crosses the channel.

Both are on-policy without replay ("we disable experience replay"), use
parameter sharing with an agent-index embedding by default, epsilon = 0.05,
gamma = 1, hard target copies every 100 batches of 32 parallel episodes,
learning rate 5e-4 and sigma = 2 (Section 6).

Release reconciliation (``iassael/learning-to-communicate`` at ``1cdfc235``,
Apache-2.0, ``code/comm.lua`` and ``code/model/Switch.lua``). Where the release
differs from the paper text, the release is reproduced:

- Optimizer: the release calls plain ``optim.rmsprop`` (alpha 0.99, eps 1e-8,
  no momentum); the paper states RMSProp with momentum 0.95.
- Gradients are normalised by ``n_agents * batch_size`` and clamped elementwise
  to [-10, 10], not norm-clipped.
- DRU placement: the sender adds Gaussian noise to the raw message, and the
  receiver applies the logistic, then batch normalisation, a linear layer and
  ReLU. This equals the paper's DRU(m) = Logistic(N(m, sigma)) followed by the
  message MLP with batch normalisation (Section 6.1).
- Execution threshold: the release binarises at ``m > 0.5`` (mapped to +/-20
  before the receiver's logistic), whereas the paper writes 1{m > 0}. The
  threshold is configurable; 0.5 is the default.
- RIAL messages are one-hot over 2^bits options, enter the message MLP without
  logistic, batch normalisation or ReLU, and the previous message is embedded
  alongside the previous action. With ``average_q_targets`` (release default),
  each head's bootstrap target is the mean of the two heads' next-step maxima
  when the agent can still both act and send.
- The RNN weights are initialised uniformly in [-0.08, 0.08].

Deliberate departures from the release, all outside the learning rule:

- Execution batch norm uses running statistics (``eval()``). The release keeps the
  online C-Net in training mode during test episodes, so its message batch norm
  normalises across the parallel test episodes, which couples supposedly
  decentralised agents in different episodes.
- The target C-Nets follow the online train/eval mode, so in training their message
  batch norm uses batch statistics. The release runs the target in evaluation mode
  but copies parameters only, leaving its running statistics at mean 0, variance 1.
- ``torch.nn.GRU`` applies the reset gate after the hidden-to-hidden product,
  r * (W h + b); the release's ``module/GRU.lua`` applies it before, W (r * h).
  Both are standard GRU formulations.
- The task encoder is a linear layer with bias; for the switch riddle's one-hot
  observations it equals the release's lookup table up to that bias.

Adaptation: the release hard-codes the switch riddle's channel. Here each
receiver's incoming messages are selected by a per-step ``communication_mask``
(receiver x sender, default: everyone hears everyone else) and either summed
into one slot (the release's limited channel, which hides who spoke) or
concatenated per sender (broadcast). The release's ``model_bn`` output layer
switch is unused there and omitted here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EXECUTION_MESSAGE_MAGNITUDE = 20.0  # release: (m > 0.5 ? +1 : -1) * 20 before the logistic


@dataclass(frozen=True)
class DIALConfig:
    mode: str = "dial"                       # "dial" or "rial"
    message_bits: int = 1                    # DIAL: real-valued width; RIAL: 2**bits messages
    hidden_dim: int = 128
    rnn_layers: int = 2
    task_hidden_dims: tuple[int, ...] = ()   # () = one linear layer (a lookup for one-hot inputs)
    parameter_sharing: bool = True
    message_aggregation: str = "concat"      # "concat" per sender, or "sum" into one slot
    noise_std: float = 2.0                   # sigma (Section 6; Appendix C)
    execution_threshold: float = 0.5         # release value; the paper writes 0
    batch_norm: bool = True
    action_aware: bool = True
    average_q_targets: bool = True           # RIAL only
    gamma: float = 1.0
    epsilon: float = 0.05
    learning_rate: float = 5e-4
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-8
    gradient_clip_value: float = 10.0
    target_update_interval: int = 100        # in batches (training iterations)
    rnn_init_range: float = 0.08

    def __post_init__(self) -> None:
        if self.mode not in {"dial", "rial"}:
            raise ValueError("mode must be 'dial' or 'rial'")
        if self.message_aggregation not in {"concat", "sum"}:
            raise ValueError("message_aggregation must be 'concat' or 'sum'")
        if self.message_bits < 1:
            raise ValueError("message_bits must be positive")

    @property
    def message_width(self) -> int:
        """Width of one sender's message on the channel."""
        return self.message_bits if self.mode == "dial" else 2 ** self.message_bits


class DRU(nn.Module):
    """Sender half of the discretise/regularise unit (Section 5.2, Appendix A).

    Training adds channel noise to the raw message; execution maps it to +/-20 so
    that the receiver's logistic yields an (almost exactly) binary value.
    """

    def __init__(self, noise_std: float, threshold: float) -> None:
        super().__init__()
        self.noise_std, self.threshold = noise_std, threshold

    def forward(self, message: Tensor, *, discretize: bool) -> Tensor:
        if discretize:
            positive = message > self.threshold
            magnitude = torch.full_like(message, EXECUTION_MESSAGE_MAGNITUDE)
            return torch.where(positive, magnitude, -magnitude)
        if self.noise_std > 0:
            return message + torch.randn_like(message) * self.noise_std
        return message


class CNet(nn.Module):
    """Section 6.1 C-Net: summed input embeddings, stacked GRU, Q and message heads."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, n_senders: int,
                 config: DIALConfig) -> None:
        super().__init__()
        hidden, width = config.hidden_dim, config.message_width
        self.config, self.action_dim = config, action_dim
        layers: list[nn.Module] = []
        previous = obs_dim
        for size in config.task_hidden_dims:
            layers += [nn.Linear(previous, size), nn.ReLU()]
            previous = size
        layers.append(nn.Linear(previous, hidden))
        self.task_encoder = nn.Sequential(*layers)
        self.agent_lookup = nn.Embedding(n_agents, hidden)
        self.action_lookup = nn.Embedding(action_dim + 1, hidden) if config.action_aware else None
        self.message_action_lookup = (
            nn.Embedding(width + 1, hidden)
            if config.action_aware and config.mode == "rial" else None
        )
        self.incoming_width = width * (n_senders if config.message_aggregation == "concat" else 1)
        message_layers: list[nn.Module] = []
        if config.mode == "dial":
            message_layers.append(nn.Sigmoid())
            if config.batch_norm:
                message_layers.append(nn.BatchNorm1d(self.incoming_width))
        message_layers.append(nn.Linear(self.incoming_width, hidden))
        if config.mode == "dial":
            message_layers.append(nn.ReLU())
        self.message_mlp = nn.Sequential(*message_layers)
        self.rnn = nn.GRU(hidden, hidden, num_layers=config.rnn_layers, batch_first=True)
        for parameter in self.rnn.parameters():
            nn.init.uniform_(parameter, -config.rnn_init_range, config.rnn_init_range)
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim + width),
        )

    def forward(
        self, obs: Tensor, incoming: Tensor, hidden: Tensor, agent_index: Tensor,
        previous_action: Tensor, previous_message: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(q_values, message_head, next_hidden)`` for one time step.

        ``previous_action`` and ``previous_message`` use index 0 for "none" and
        ``k + 1`` for choice ``k``, as in the release's lookup tables.
        """
        z = self.task_encoder(obs) + self.agent_lookup(agent_index) + self.message_mlp(incoming)
        if self.action_lookup is not None:
            z = z + self.action_lookup(previous_action)
        if self.message_action_lookup is not None and previous_message is not None:
            z = z + self.message_action_lookup(previous_message)
        output, next_hidden = self.rnn(z.unsqueeze(1), hidden)
        head = self.output(output.squeeze(1))
        return head[:, : self.action_dim], head[:, self.action_dim:], next_hidden


@dataclass
class _EpisodeRecord:
    chosen_q: list[Tensor] = field(default_factory=list)           # (B, N), online, with grad
    chosen_message_q: list[Tensor] = field(default_factory=list)   # RIAL (B, N)
    message_taken: list[Tensor] = field(default_factory=list)      # RIAL (B, N) float
    target_action_max: list[Tensor] = field(default_factory=list)  # (B, N), target
    target_message_max: list[Tensor] = field(default_factory=list)  # RIAL (B, N), target
    rewards: list[Tensor] = field(default_factory=list)            # (B,)
    terminated: list[Tensor] = field(default_factory=list)         # (B,)
    active: list[Tensor] = field(default_factory=list)             # (B,)


class DIALAgent(nn.Module):
    """Complete DIAL/RIAL learner: C-Nets, channel, epsilon-greedy control and DQN update.

    One training iteration processes a batch of parallel episodes on-policy:
    ``begin_episode`` -> per step ``act`` then ``observe`` -> ``finish_episode``.
    The autograd graph spans the whole batch of episodes so that DIAL's message
    gradients reach the senders, exactly as Algorithm 1's backward pass does.
    """

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 config: DIALConfig | None = None) -> None:
        super().__init__()
        self.config = config or DIALConfig()
        self.n_agents, self.obs_dim, self.action_dim = n_agents, obs_dim, action_dim
        n_networks = 1 if self.config.parameter_sharing else n_agents
        self.networks = nn.ModuleList(
            CNet(n_agents, obs_dim, action_dim, n_agents, self.config) for _ in range(n_networks)
        )
        self.target_networks = copy.deepcopy(self.networks)
        for parameter in self.target_networks.parameters():
            parameter.requires_grad_(False)
        self.dru = DRU(self.config.noise_std, self.config.execution_threshold)
        self.optimizer = torch.optim.RMSprop(
            self.networks.parameters(), lr=self.config.learning_rate,
            alpha=self.config.rmsprop_alpha, eps=self.config.rmsprop_eps,
        )
        self.iterations = 0
        self._record: _EpisodeRecord | None = None
        self._state: dict[str, Tensor] = {}

    # ------------------------------------------------------------------ control
    def network(self, agent: int, *, target: bool = False) -> CNet:
        networks = self.target_networks if target else self.networks
        return networks[0 if self.config.parameter_sharing else agent]

    def begin_episode(self, batch_size: int, device: torch.device | None = None) -> None:
        """Reset hidden states, the channel and last-action inputs for a batch."""
        device = device or next(self.parameters()).device
        if self.training and self.config.batch_norm and self.config.mode == "dial" and batch_size < 2:
            raise ValueError("training with batch normalisation needs at least two parallel episodes")
        n, width, layers = self.n_agents, self.config.message_width, self.config.rnn_layers
        zeros_hidden = torch.zeros(n, layers, batch_size, self.config.hidden_dim, device=device)
        zeros_messages = torch.zeros(batch_size, n, width, device=device)
        self._state = {
            "hidden": zeros_hidden, "target_hidden": zeros_hidden.clone(),
            "sent": zeros_messages, "target_sent": zeros_messages.clone(),
            "previous_action": torch.zeros(batch_size, n, dtype=torch.long, device=device),
            "previous_message": torch.zeros(batch_size, n, dtype=torch.long, device=device),
        }
        self._record = _EpisodeRecord() if self.training else None

    def act(
        self, obs: Tensor, available_actions: Tensor | None = None, *,
        communication_mask: Tensor | None = None, message_senders: Tensor | None = None,
        explore: bool | None = None,
    ) -> Tensor:
        """Choose environment actions ``(B, N)`` and emit this step's messages.

        ``obs`` is ``(B, N, obs_dim)``. ``communication_mask[b, r, s]`` selects
        which of last step's messages receiver ``r`` hears now (default: all but
        its own). ``message_senders[b, i]`` marks agents allowed to send this step
        (default: all). Exploration defaults to on in training mode only.
        """
        if not self._state:
            raise RuntimeError("call begin_episode() before act()")
        batch, n = obs.shape[:2]
        device = obs.device
        explore = self.training if explore is None else explore
        available = (torch.ones(batch, n, self.action_dim, device=device)
                     if available_actions is None else available_actions.to(device).float())
        mask = self._mask(communication_mask, batch, device)
        senders = (torch.ones(batch, n, device=device) if message_senders is None
                   else message_senders.to(device).float())
        incoming = self._route(self._state["sent"], mask)
        q_values, message_head, hidden = self._forward_all(
            obs, incoming, self._state["hidden"], target=False,
        )
        actions = self._epsilon_greedy(q_values, available, explore)
        record = self._record
        if record is not None:
            record.chosen_q.append(q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1))

        if self.config.mode == "dial":
            sent = self.dru(message_head, discretize=not self.training) * senders.unsqueeze(-1)
            message_index = torch.zeros_like(actions)
        else:
            message_q = message_head
            all_messages = torch.ones_like(message_q)
            chosen = self._epsilon_greedy(message_q, all_messages, explore)
            sent = F.one_hot(chosen, self.config.message_width).float() * senders.unsqueeze(-1)
            message_index = torch.where(senders > 0, chosen + 1, torch.zeros_like(chosen))
            if record is not None:
                record.chosen_message_q.append(message_q.gather(-1, chosen.unsqueeze(-1)).squeeze(-1))
                record.message_taken.append(senders)

        if record is not None:
            self._target_step(obs, available, mask, senders, actions)
        self._state.update(
            hidden=hidden, sent=sent,
            previous_action=actions + 1,
            previous_message=message_index,
        )
        return actions

    def observe(self, rewards: Tensor, terminated: Tensor, active: Tensor) -> None:
        """Record the team reward ``(B,)``, termination and which episodes were still running."""
        if self._record is None:
            return
        device = self._state["hidden"].device
        self._record.rewards.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
        self._record.terminated.append(torch.as_tensor(terminated, dtype=torch.float32, device=device))
        self._record.active.append(torch.as_tensor(active, dtype=torch.float32, device=device))

    def finish_episode(
        self, final_obs: Tensor | None = None, final_available_actions: Tensor | None = None, *,
        final_communication_mask: Tensor | None = None, final_message_senders: Tensor | None = None,
    ) -> dict[str, float]:
        """Compute the DQN loss over the batch, update, and refresh targets on schedule.

        The final observation is only needed when an episode was cut by a time
        limit rather than terminated, to bootstrap its last step.
        """
        record = self._record
        if record is None:
            raise RuntimeError("finish_episode() requires training mode and begin_episode()")
        if len(record.rewards) != len(record.chosen_q):
            raise RuntimeError("call observe() once after every act()")
        steps = len(record.chosen_q)
        if steps == 0:
            raise RuntimeError("no steps recorded")
        last_needs_bootstrap = bool(
            (record.active[-1] * (1.0 - record.terminated[-1])).sum() > 0
        )
        if last_needs_bootstrap:
            if final_obs is None:
                raise ValueError("final_obs is required to bootstrap truncated episodes")
            batch = final_obs.shape[0]
            device = final_obs.device
            available = (torch.ones(batch, self.n_agents, self.action_dim, device=device)
                         if final_available_actions is None
                         else final_available_actions.to(device).float())
            mask = self._mask(final_communication_mask, batch, device)
            senders = (torch.ones(batch, self.n_agents, device=device)
                       if final_message_senders is None else final_message_senders.to(device).float())
            self._target_step(final_obs, available, mask, senders, None)
        else:
            zeros = torch.zeros_like(record.target_action_max[-1])
            record.target_action_max.append(zeros)
            if self.config.mode == "rial":
                record.target_message_max.append(zeros)

        loss, td_squared, message_td_squared = self._loss(record, steps)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.networks.parameters(), self.config.gradient_clip_value)
        self.optimizer.step()
        self.iterations += 1
        if self.iterations % self.config.target_update_interval == 0:
            self.update_targets()
        self._record = None
        self._state = {}
        metrics = {"loss": float(loss.detach()), "td_error_sq": td_squared}
        if self.config.mode == "rial":
            metrics["message_td_error_sq"] = message_td_squared
        return metrics

    def update_targets(self) -> None:
        self.target_networks.load_state_dict(self.networks.state_dict())

    # ---------------------------------------------------------------- internals
    def _mask(self, mask: Tensor | None, batch: int, device: torch.device) -> Tensor:
        if mask is None:
            default = 1.0 - torch.eye(self.n_agents, device=device)
            return default.expand(batch, -1, -1)
        return mask.to(device).float()

    def _route(self, sent: Tensor, mask: Tensor) -> Tensor:
        """Deliver last step's messages ``(B, N_send, W)`` to receivers ``(B, N_recv, *)``."""
        selected = mask.unsqueeze(-1) * sent.unsqueeze(1)            # (B, R, S, W)
        if self.config.message_aggregation == "sum":
            return selected.sum(dim=2)
        return selected.flatten(start_dim=2)

    def _forward_all(self, obs: Tensor, incoming: Tensor, hidden: Tensor, *, target: bool
                     ) -> tuple[Tensor, Tensor, Tensor]:
        batch = obs.shape[0]
        q_list, message_list, hidden_list = [], [], []
        previous_action = self._state["previous_action"]
        previous_message = self._state["previous_message"]
        for agent in range(self.n_agents):
            index = torch.full((batch,), agent, dtype=torch.long, device=obs.device)
            q, message, next_hidden = self.network(agent, target=target)(
                obs[:, agent], incoming[:, agent], hidden[agent].contiguous(), index,
                previous_action[:, agent], previous_message[:, agent],
            )
            q_list.append(q)
            message_list.append(message)
            hidden_list.append(next_hidden)
        return torch.stack(q_list, 1), torch.stack(message_list, 1), torch.stack(hidden_list, 0)

    def _epsilon_greedy(self, q_values: Tensor, available: Tensor, explore: bool) -> Tensor:
        masked = q_values.detach().masked_fill(available <= 0, -torch.inf)
        greedy = masked.argmax(-1)
        if not explore or self.config.epsilon <= 0:
            return greedy
        random_scores = torch.rand_like(masked).masked_fill(available <= 0, -1.0)
        random_choice = random_scores.argmax(-1)
        flip = torch.rand(greedy.shape, device=greedy.device) < self.config.epsilon
        return torch.where(flip, random_choice, greedy)

    @torch.no_grad()
    def _target_step(self, obs: Tensor, available: Tensor, mask: Tensor, senders: Tensor,
                     actions: Tensor | None) -> None:
        """Run the target C-Nets alongside the online ones.

        DIAL targets exchange their own noisy messages (release: ``comm_target``);
        RIAL targets hear the discrete messages the online agents actually sent.
        """
        record = self._record
        channel = self._state["target_sent"] if self.config.mode == "dial" else self._state["sent"]
        incoming = self._route(channel, mask)
        q_values, message_head, hidden = self._forward_all(
            obs, incoming, self._state["target_hidden"], target=True,
        )
        action_max = q_values.masked_fill(available <= 0, -torch.inf).amax(-1)
        action_max = torch.nan_to_num(action_max, neginf=0.0)
        record.target_action_max.append(action_max)
        if self.config.mode == "dial":
            target_sent = self.dru(message_head, discretize=False) * senders.unsqueeze(-1)
        else:
            message_max = message_head.amax(-1)
            record.target_message_max.append(torch.where(senders > 0, message_max, action_max))
            target_sent = self._state["target_sent"]
        if actions is not None:
            self._state.update(target_hidden=hidden, target_sent=target_sent)

    def _loss(self, record: _EpisodeRecord, steps: int) -> tuple[Tensor, float, float]:
        gamma = self.config.gamma
        batch = record.chosen_q[0].shape[0]
        total = record.chosen_q[0].new_zeros(())
        td_squared, message_td_squared, count = 0.0, 0.0, 0.0
        for t in range(steps):
            reward = record.rewards[t].unsqueeze(-1)
            continuing = (1.0 - record.terminated[t]).unsqueeze(-1)
            weight = record.active[t].unsqueeze(-1).expand_as(record.chosen_q[t])
            next_action = record.target_action_max[t + 1]
            if self.config.mode == "rial":
                next_message = record.target_message_max[t + 1]
                if self.config.average_q_targets:
                    sending = record.message_taken[t]
                    averaged = 0.5 * (next_action + next_message)
                    action_bootstrap = torch.where(sending > 0, averaged, next_action)
                    message_bootstrap = averaged
                else:
                    action_bootstrap, message_bootstrap = next_action, next_message
            else:
                action_bootstrap = next_action
            target = reward + gamma * continuing * action_bootstrap
            td = (target - record.chosen_q[t]) * weight
            total = total + 0.5 * td.square().sum()
            td_squared += float(td.detach().square().sum())
            count += float(weight.sum())
            if self.config.mode == "rial":
                message_weight = record.active[t].unsqueeze(-1) * record.message_taken[t]
                message_target = reward + gamma * continuing * message_bootstrap
                message_td = (message_target - record.chosen_message_q[t]) * message_weight
                total = total + 0.5 * message_td.square().sum()
                message_td_squared += float(message_td.detach().square().sum())
        loss = total / (self.n_agents * batch)
        return loss, td_squared / max(count, 1.0), message_td_squared / max(count, 1.0)


__all__ = ["DRU", "CNet", "DIALAgent", "DIALConfig"]
