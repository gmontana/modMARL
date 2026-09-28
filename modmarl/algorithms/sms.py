"""SMS implementation.

Original paper:
Di Xue, Lei Yuan, Zongzhang Zhang, Yang Yu. "Efficient Multi-Agent Communication via
Shapley Message Value." IJCAI 2022, pp. 578-584.
Official code: https://github.com/DiXue98/SMS (Apache-2.0), reference commit
e01327924ee197b06fb5012e88ab036e1196e826 (the repository has exactly one commit). The
paper underspecifies architecture and training, so the released code is the fidelity
reference. The single-process trainer holds the policy fixed while collecting the release's
eight trajectories, preserving its parallel runner's data-to-update ratio exactly.

DOP-style actor-critic where communication is priced by cooperative game theory. Each
agent's policy sees its controller input (observation and agent identity) plus, in the
policy head only, a vector of teammate messages: sender j encodes its observation and
identity into n recipient-specific slices c_{j->i}, and receiver i concatenates them by
sender slot under a binary mask F with F_ii = 0, so pruned slots are exact zeros (paper
3.1). Messages never enter the GRU, so one replayed hidden is valid for every
counterfactual message subset -- which is what makes Shapley estimation affordable.

The centralized critic is the released dueling form: Q_i = V(inputs) + [A(inputs, a) -
mean_{a'} A(inputs, onehot(a'))] over inputs o_i (+) state (+) onehot(id_i), taking action
probability vectors. Two critics are kept and min(Q1, Q2) feeds both TD targets and
Shapley evaluation. They are linearly decomposed (paper Eq. 2) as Q_tot = sum_i k_i(s)Q_i
with non-negative hypernet coefficients normalised to sum one.

A message's worth is its Shapley Message Value (Eq. 4-6), estimated by permutation
sampling (Eq. 12): H random coalition orderings, prefix policy evaluations under the fixed
replayed hidden, backward-differenced into marginals. A selector MLP is regressed onto the
k_i(s)-scaled labels (Eq. 7, 11); at execution, after t_selector env steps, agent i
requests messages only from teammates with positive predicted SMV, so no Shapley
computation happens at execution.

Training follows the released two-buffer scheme (Eq. 8-9). The off-policy stream runs
first on a uniform replay sample and trains only the critics and mixer, relabelling
messages under the target policy; the on-policy stream then trains critics and mixer on
TD(lambda), the actor on Eq. 10, and the selector on the SMV labels. The policy is trained
under full communication with message dropout (p = 0.5) so each message stays useful alone
(paper 3.4).

Divergences from the release, each pinned by a test where it is observable:
- Controller inputs omit the last action. Both published environment configs set the
  top-level ``obs_last_action: False``; ``default.yaml``'s ``True`` is the parser default.
- Selector labels are index-aligned. ``sms_learner.py`` multiplies a time-major stack of
  Shapley values by a batch-major ``k(s)``, pairing most rows with a different timestep's
  coefficient; paper Equations 5 and 7 intend them aligned.
- Off-policy relabelling applies the selector gate and injects the Gaussian message noise,
  as ``sms_controller.py`` does once ``t_env`` reaches ``t_selector``. Relabelling under
  full, noiseless communication would score a policy the agent never runs.
- The mixer's hypernets are the release's bare affine maps; its ``hypernet_embed`` key is
  read by no upstream code.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common.nn import build_mlp


def _agent_ids(batch: int, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """One-hot agent identity block (B, n, n)."""
    return torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch, n, n)


class SMSAgent(nn.Module):
    """SMS policy: GRU over the controller inputs; peer-to-peer recipient-specific
    messages that feed the policy head only; the Shapley selector gate.

    Controller inputs are ``obs (+) onehot(last_action) (+) onehot(agent_id)``, with the
    two optional blocks controlled by the released environment configuration. The message
    content is encoded from ``obs (+) onehot(id)``.
    The mask convention everywhere is (B, n, n) with row = receiver, column = sender,
    and F_ii = 0 enforced by ``forward`` regardless of the mask handed in.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        msg_dim: int = 8,
        msg_hidden_dim: int = 32,
        selector_hidden_dim: int = 32,
        *,
        use_rnn: bool = False,
        include_last_action: bool = False,
        include_agent_id: bool = True,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.use_rnn = use_rnn
        self.include_last_action = include_last_action
        self.include_agent_id = include_agent_id
        input_dim = obs_dim + (action_dim if include_last_action else 0) + (n_agents if include_agent_id else 0)
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.msg_encoder = build_mlp(obs_dim + n_agents, [msg_hidden_dim], n_agents * msg_dim)
        self.selector = build_mlp(input_dim, [selector_hidden_dim], n_agents)
        head_input = (hidden_dim if use_rnn else input_dim) + n_agents * msg_dim
        self.head = nn.Linear(head_input, action_dim)

    def init_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.n_agents, self.hidden_dim, device=device)

    def build_inputs(self, obs: Tensor, last_action: Tensor | None = None) -> Tensor:
        """Controller inputs (B, n, input_dim) = obs (+) last action (+) agent id."""
        pieces = [obs]
        if self.include_last_action:
            if last_action is None:
                last_action = torch.zeros(*obs.shape[:2], self.action_dim, device=obs.device, dtype=obs.dtype)
            pieces.append(last_action.to(dtype=obs.dtype))
        if self.include_agent_id:
            pieces.append(_agent_ids(obs.shape[0], self.n_agents, obs.device, obs.dtype))
        return torch.cat(pieces, dim=-1)

    def build_messages(self, obs: Tensor, *, noise: bool = False) -> Tensor:
        """All pairwise message contents: (B, n, obs_dim) -> (B, n, n, msg_dim) with
        [b, i, j] = c_{j->i}, sender j's recipient-specific slice for receiver i, encoded
        from sender j's observation and identity. ``noise`` adds the released rollout-time
        Gaussian perturbation (kept off for critic/actor/Shapley evaluation)."""
        batch, n, _ = obs.shape
        ids = _agent_ids(batch, n, obs.device, obs.dtype)
        content = self.msg_encoder(torch.cat([obs, ids], dim=-1)).view(batch, n, n, self.msg_dim).transpose(1, 2)
        if noise:
            content = content + torch.randn_like(content)
        return content

    def selector_scores(self, obs: Tensor, last_action: Tensor | None = None) -> Tensor:
        """Predicted SMV of every sender for every receiver: (B, n, n), row = receiver,
        column = sender; the diagonal is meaningless and ignored by all callers."""
        return self.selector(self.build_inputs(obs, last_action))

    def comm_mask(
        self, obs: Tensor, last_action: Tensor | None = None, *, selector_on: bool, dropout_p: float = 0.0
    ) -> Tensor:
        """Acting-time communication mask (B, n, n) in {0, 1}: full-comm 1 - I, times the
        selector gate 1[selector > 0] once t_selector has passed, times iid message
        dropout 1[U > p] during training rollouts and actor evaluation."""
        batch, n, _ = obs.shape
        mask = 1.0 - torch.eye(n, device=obs.device).expand(batch, n, n)
        if selector_on:
            with torch.no_grad():
                mask = mask * (self.selector_scores(obs, last_action) > 0).to(mask.dtype)
        if dropout_p > 0.0:
            mask = mask * (torch.rand_like(mask) > dropout_p).to(mask.dtype)
        return mask

    def forward(
        self, obs: Tensor, hidden: Tensor, mask: Tensor, last_action: Tensor | None = None, *, noise: bool = False
    ) -> tuple[Tensor, Tensor]:
        """One policy step: (B, n, obs_dim), (B, n, H), mask (B, n, n) -> (logits
        (B, n, A), hidden'). The GRU sees only the controller inputs — messages enter the
        head, so hidden' is identical for every mask."""
        batch, n, _ = obs.shape
        inputs = self.build_inputs(obs, last_action)
        x = torch.relu(self.fc1(inputs)).reshape(batch * n, self.hidden_dim)
        new_hidden = self.gru(x, hidden.reshape(batch * n, self.hidden_dim)).view(batch, n, self.hidden_dim)
        gate = mask * (1.0 - torch.eye(n, device=obs.device))            # F_ii = 0 invariant
        received = (gate.unsqueeze(-1) * self.build_messages(obs, noise=noise)).reshape(batch, n, n * self.msg_dim)
        trunk = new_hidden if self.use_rnn else inputs
        return self.head(torch.cat([trunk, received], dim=-1)), new_hidden

    def sync_selector(self, trained_selector: nn.Module) -> None:
        """Hard-copy the learner-side selector into the acting agent (the execution gate
        changes only at target-update time)."""
        self.selector.load_state_dict(trained_selector.state_dict())


class LinearDecompositionMixer(nn.Module):
    """DOP mixer (paper Eq. 2): Q_tot = sum_i k_i(s) Q_i with hypernet coefficients
    k(s) = |W1(s)| |W2(s)| normalised so sum_i k_i(s) = 1 and k >= 0. The bias b(s) is
    omitted, as the official forward does for both the TD and the actor loss.

    ``k`` is exposed separately — the SMV labels need it.
    forward(agent_qs (B, n), state (B, state_dim)) -> (B,).
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        # The released `DopMixer` uses bare affine maps, not hypernet MLPs; its
        # `hypernet_embed: 64` config key is read by no code upstream.
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)

    def k(self, state: Tensor) -> Tensor:
        """Per-agent mixing coefficients: (B, state_dim) -> (B, n), non-negative, rows
        summing to one (the normalised rank-1 |W1||W2| product)."""
        batch = state.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.embed_dim)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        coeff = torch.bmm(w1, w2).view(batch, self.n_agents)
        return coeff / coeff.sum(dim=1, keepdim=True)

    def forward(self, agent_qs: Tensor, state: Tensor) -> Tensor:
        return (self.k(state) * agent_qs).sum(dim=1)


class SMSCritic(nn.Module):
    """Per-agent dueling centralized critic (the official ``FMACDuelingCritic``):

        Q_i(inputs, a) = V(inputs) + [A(inputs, a) - mean_{a'} A(inputs, onehot(a'))]

    with inputs = o_i (+) state (+) onehot(id_i) and ``a`` an action PROBABILITY vector
    (one-hot for TD, softmax for the actor and message evaluation). Two of these are kept
    by the learner and combined with min(Q1, Q2).

    forward(obs (B, n, obs_dim), state (B, state_dim), probs (B, n, A)) -> (B, n, 1).
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        include_agent_id: bool = True,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.include_agent_id = include_agent_id
        input_shape = obs_dim + state_dim + (n_agents if include_agent_id else 0)
        self.input_shape = input_shape
        self.advantage = build_mlp(input_shape + action_dim, [hidden_dim, hidden_dim], 1)
        self.value = build_mlp(input_shape, [hidden_dim, hidden_dim], 1)

    def _inputs(self, obs: Tensor, state: Tensor) -> Tensor:
        batch, n, _ = obs.shape
        pieces = [obs, state.unsqueeze(1).expand(-1, n, -1)]
        if self.include_agent_id:
            pieces.append(_agent_ids(batch, n, obs.device, obs.dtype))
        return torch.cat(pieces, dim=-1)

    def forward(self, obs: Tensor, state: Tensor, probs: Tensor) -> Tensor:
        batch, n, _ = obs.shape
        inputs = self._inputs(obs, state)                                # (B, n, input_shape)
        a_taken = self.advantage(torch.cat([inputs, probs], dim=-1))     # (B, n, 1)
        eye = torch.eye(self.action_dim, device=obs.device, dtype=obs.dtype)
        dueling = torch.cat(
            [
                inputs.unsqueeze(-2).expand(-1, -1, self.action_dim, -1),
                eye.view(1, 1, self.action_dim, self.action_dim).expand(batch, n, -1, -1),
            ],
            dim=-1,
        )                                                                # (B, n, A, input_shape + A)
        baseline = self.advantage(dueling).mean(dim=-2)                  # (B, n, 1)
        return self.value(inputs) + a_taken - baseline


def _policy_value(
    agent: SMSAgent,
    critic1: SMSCritic,
    critic2: SMSCritic,
    obs: Tensor,
    state: Tensor,
    hidden: Tensor,
    last_action: Tensor | None,
    mask: Tensor,
    *,
    noise: bool,
) -> Tensor:
    """min(Q1, Q2) of the policy's softmax under one message mask: (B, n, 1)."""
    logits, _ = agent(obs, hidden, mask, last_action, noise=noise)
    probs = torch.softmax(logits, dim=-1)
    return torch.min(critic1(obs, state, probs), critic2(obs, state, probs))


def shapley_message_values(
    agent: SMSAgent,
    critic1: SMSCritic,
    critic2: SMSCritic,
    obs: Tensor,
    state: Tensor,
    hidden: Tensor,
    last_action: Tensor | None = None,
    *,
    sample_size: int = 2,
    noise: bool = True,
) -> Tensor:
    """Permutation-sampling SMV estimator (paper Eq. 12), vectorized over receivers,
    evaluated with the twin critics' min (as the official code does).

    ``hidden`` is the pre-step recurrent state; the policy head is message-independent, so
    the one replayed hidden serves every counterfactual subset. Per sample: one uniform
    random ordering of each receiver's coalition (the receiver pinned at position 0),
    prefix policy evaluations through the critics after unmasking one sender at a time,
    then backward differences along the ordering — each permutation yields exact marginals
    whose per-receiver sum telescopes to Q(full) - Q(empty).

    Returns (B, n, n), detached: [b, i, j] = UNSCALED Shapley marginal of sender j's
    message to receiver i — callers multiply by the mixer's k_i(s) (Eq. 5). The diagonal
    holds Q_i(empty) by construction; callers must ignore it.
    """
    batch, n, _ = obs.shape
    receiver = torch.arange(n, device=obs.device).view(1, n, 1).expand(batch, n, 1)
    uniform = (1.0 - torch.eye(n, device=obs.device)).repeat(batch, 1)
    total = torch.zeros(batch, n, n, device=obs.device)
    with torch.no_grad():
        for _ in range(sample_size):
            order = torch.multinomial(uniform, n - 1).view(batch, n, n - 1)
            mask = torch.zeros(batch, n, n, device=obs.device)
            value = torch.zeros(batch, n, n, device=obs.device)
            value.scatter_(
                2, receiver, _policy_value(
                    agent, critic1, critic2, obs, state, hidden, last_action, mask, noise=noise,
                ),
            )
            for position in range(n - 1):
                sender = order[:, :, position : position + 1]
                mask.scatter_(2, sender, 1.0)
                value.scatter_(
                    2, sender, _policy_value(
                        agent, critic1, critic2, obs, state, hidden, last_action, mask, noise=noise,
                    ),
                )
            # Backward-difference the prefix values into marginals: each unmasked sender's
            # value minus its predecessor's (the receiver's own slot — the empty-coalition
            # value — precedes position 0).
            prefix = value.gather(2, order)
            previous = torch.cat([value.gather(2, receiver), prefix[:, :, :-1]], dim=2)
            marginal = torch.zeros_like(value).scatter_(2, order, prefix - previous)
            marginal.scatter_(2, receiver, value.gather(2, receiver))
            total += marginal
    return total / sample_size


__all__ = ["LinearDecompositionMixer", "SMSAgent", "SMSCritic", "shapley_message_values"]
