from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples import train_sms
from examples.train_sms import train
from marl_envs import make_env
from modmarl.algorithms.sms import (
    LinearDecompositionMixer,
    SMSAgent,
    SMSCritic,
    shapley_message_values,
)
from modmarl.common.replay import EpisodeReplayBuffer

# Controller input width for the default test agent: obs 5 + last-action 4 + agent-id 3.
INPUT_DIM = 8          # obs(5) + agent id(3); the release feeds no last action
MSG_OFFSET = INPUT_DIM        # head input is [controller inputs, messages]; slot j at INPUT_DIM + j*msg_dim


def _agent(**overrides) -> SMSAgent:
    kwargs = dict(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16, msg_dim=2, msg_hidden_dim=8, selector_hidden_dim=8)
    kwargs.update(overrides)
    return SMSAgent(**kwargs)


def _critic(**overrides) -> SMSCritic:
    kwargs = dict(n_agents=3, obs_dim=5, state_dim=15, action_dim=4, hidden_dim=16)
    kwargs.update(overrides)
    return SMSCritic(**kwargs)


def _learner(agent: SMSAgent | None = None) -> dict:
    agent = agent if agent is not None else _agent()
    critic1, critic2 = _critic(), _critic()
    mixer = LinearDecompositionMixer(3, 15, embed_dim=8)
    return dict(
        agent=agent,
        target_agent=copy.deepcopy(agent),
        critic1=critic1,
        critic2=critic2,
        target_critic1=copy.deepcopy(critic1),
        target_critic2=copy.deepcopy(critic2),
        mixer=mixer,
        target_mixer=copy.deepcopy(mixer),
        learner_selector=copy.deepcopy(agent.selector),
        actor_opt=torch.optim.Adam(train_sms._actor_params(agent), lr=1e-3),
        critic_opt=torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()) + list(mixer.parameters()), lr=1e-3
        ),
        selector_opt=torch.optim.Adam(copy.deepcopy(agent.selector).parameters(), lr=1e-4),
    )


def _call_update(learner: dict, batch, *, off: bool = False, gamma=0.9, td_lambda=0.6, entropy_coef=0.03,
                 dropout_p=0.5, smv_sample_size=2) -> None:
    train_sms._update(
        learner["agent"], learner["target_agent"], learner["critic1"], learner["critic2"],
        learner["target_critic1"], learner["target_critic2"], learner["mixer"], learner["target_mixer"],
        learner["learner_selector"], learner["actor_opt"], learner["critic_opt"], learner["selector_opt"],
        batch, gamma, td_lambda, entropy_coef, dropout_p, smv_sample_size, off=off,
    )


def _episode_batch(horizon: int, length: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    buffer = EpisodeReplayBuffer(capacity=4, horizon=horizon, n_agents=3, obs_dim=5)
    buffer.add_episode(
        obs=rng.standard_normal((length + 1, 3, 5)).astype(np.float32),
        actions=rng.integers(0, 4, (length, 3)),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=np.zeros(length, dtype=np.float32),
    )
    return buffer.sample(1, torch.device("cpu"))


def test_message_layout_and_mask_convention() -> None:
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 16)

    # Sender content: msg[b, i, j] = c_{j->i} depends on sender j's observation only
    # (the agent-identity block appended to the encoder input is constant).
    msg = agent.build_messages(obs)
    assert tuple(msg.shape) == (2, 3, 3, 2)
    shifted = obs.clone()
    shifted[:, 1] += 1.0
    msg_shifted = agent.build_messages(shifted)
    assert torch.equal(msg_shifted[:, :, [0, 2]], msg[:, :, [0, 2]])
    assert not torch.allclose(msg_shifted[:, :, 1], msg[:, :, 1])

    # F_ii = 0 invariant: an all-ones mask acts exactly like 1 - I.
    ones = torch.ones(2, 3, 3)
    no_diag = (1.0 - torch.eye(3)).expand(2, 3, 3)
    full_logits, _ = agent(obs, hidden, ones)
    no_diag_logits, _ = agent(obs, hidden, no_diag)
    assert torch.equal(full_logits, no_diag_logits)

    # Cutting mask column j removes exactly sender slot j of every receiver's concatenated
    # vector: with the linear head, the logits change by W_j @ c_{j->i} and nothing else.
    j = 1
    cut = ones.clone()
    cut[:, :, j] = 0.0
    cut_logits, _ = agent(obs, hidden, cut)
    w_j = agent.head.weight[:, MSG_OFFSET + j * 2 : MSG_OFFSET + (j + 1) * 2]     # (A, msg_dim)
    expected = torch.einsum("bid,ad->bia", msg[:, :, j], w_j)
    expected[:, j] = 0.0
    assert torch.allclose(full_logits - cut_logits, expected, atol=1e-6)


def test_dueling_critic_equals_value_plus_centered_advantage() -> None:
    # The official FMACDuelingCritic form: Q = V(inputs) + A(inputs, a) - mean_a' A(inputs, onehot(a')).
    torch.manual_seed(0)
    critic = _critic()
    obs, state = torch.randn(2, 3, 5), torch.randn(2, 15)
    probs = torch.softmax(torch.randn(2, 3, 4), dim=-1)

    q = critic(obs, state, probs)
    inputs = critic._inputs(obs, state)
    a_taken = critic.advantage(torch.cat([inputs, probs], dim=-1))
    eye = torch.eye(4)
    onehot_adv = torch.stack(
        [critic.advantage(torch.cat([inputs, eye[a].view(1, 1, 4).expand(2, 3, 4)], dim=-1)) for a in range(4)],
        dim=-2,
    )                                                                    # (2, 3, 4, 1)
    expected = critic.value(inputs) + a_taken - onehot_adv.mean(dim=-2)
    assert tuple(q.shape) == (2, 3, 1)
    assert torch.allclose(q, expected, atol=1e-6)


def test_controller_inputs_exclude_last_action_and_include_agent_id() -> None:
    # Both published env configs set the top-level `obs_last_action: False`;
    # `default.yaml`'s True is only the parser default.
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    hidden = agent.init_state(2, torch.device("cpu"))
    no_diag = (1.0 - torch.eye(3)).expand(2, 3, 3)

    # Controller input width and message slot position.
    assert agent.include_last_action is False
    assert agent.input_dim == INPUT_DIM
    assert agent.head.in_features == INPUT_DIM + 3 * 2                   # [inputs, messages]

    # The previous action is not a policy input, so it cannot move the logits.
    last_a = torch.zeros(2, 3, 4)
    last_a[:, :, 0] = 1.0
    last_b = torch.zeros(2, 3, 4)
    last_b[:, :, 1] = 1.0
    logits_a, _ = agent(obs, hidden, no_diag, last_a)
    logits_b, _ = agent(obs, hidden, no_diag, last_b)
    assert torch.allclose(logits_a, logits_b)

    # Opting in widens the controller input and makes it matter again.
    with_action = _agent(include_last_action=True)
    assert with_action.input_dim == INPUT_DIM + 4
    hidden_wa = with_action.init_state(2, torch.device("cpu"))
    wa_a, _ = with_action(obs, hidden_wa, no_diag, last_a)
    wa_b, _ = with_action(obs, hidden_wa, no_diag, last_b)
    assert not torch.allclose(wa_a, wa_b)

    # An agent whose observations coincide is still distinguishable via the identity block.
    same_obs = torch.randn(1, 1, 5).expand(1, 3, 5).contiguous()
    inputs = agent.build_inputs(same_obs)
    assert not torch.allclose(inputs[:, 0], inputs[:, 1])               # differ only in the id block

    # Disabling the extra features drops them from the input width.
    plain = _agent(include_last_action=False, include_agent_id=False)
    assert plain.input_dim == 5


def test_feedforward_head_ignores_hidden_while_recurrent_head_uses_it() -> None:
    torch.manual_seed(0)
    obs = torch.randn(2, 3, 5)
    mask = (1.0 - torch.eye(3)).expand(2, 3, 3)

    ff = _agent(use_rnn=False)
    assert ff.head.in_features == INPUT_DIM + 3 * 2
    logits_a, _ = ff(obs, torch.zeros(2, 3, 16), mask)
    logits_b, _ = ff(obs, torch.randn(2, 3, 16), mask)
    assert torch.equal(logits_a, logits_b)                             # head is feedforward over inputs+messages

    rnn = _agent(use_rnn=True)
    assert rnn.head.in_features == 16 + 3 * 2
    logits_c, _ = rnn(obs, torch.zeros(2, 3, 16), mask)
    logits_d, _ = rnn(obs, torch.randn(2, 3, 16), mask)
    assert not torch.equal(logits_c, logits_d)                        # head consumes the GRU hidden


def test_useless_message_gets_zero_smv() -> None:
    # Zero the policy-head weights for sender j's slot: every policy is provably
    # independent of j's message, so each marginal involving j differences two identical
    # evaluations — exact zero for any sampled permutations, no tolerance.
    torch.manual_seed(0)
    agent = _agent()
    critic1, critic2 = _critic(), _critic()
    j = 1
    with torch.no_grad():
        agent.head.weight[:, MSG_OFFSET + j * 2 : MSG_OFFSET + (j + 1) * 2] = 0.0
    obs, state, hidden = torch.randn(4, 3, 5), torch.randn(4, 15), torch.randn(4, 3, 16)

    smv = shapley_message_values(agent, critic1, critic2, obs, state, hidden, sample_size=3, noise=False)
    assert torch.all(smv[:, [0, 2], j] == 0.0)
    assert not torch.all(smv == 0.0)


def test_policy_value_uses_min_of_twin_critics() -> None:
    # The Shapley evaluation prices messages with min(Q1, Q2) (official). A constant
    # offset would cancel in the SMV differences, so assert the min at the value level.
    from modmarl.algorithms.sms import _policy_value

    torch.manual_seed(0)
    agent = _agent()
    critic1 = _critic()
    critic2 = copy.deepcopy(critic1)
    with torch.no_grad():
        critic2.value[-1].bias += 100.0                                # critic2 strictly dominates
    obs, state, hidden = torch.randn(4, 3, 5), torch.randn(4, 15), torch.randn(4, 3, 16)
    mask = (1.0 - torch.eye(3)).expand(4, 3, 3)

    v_min = _policy_value(agent, critic1, critic2, obs, state, hidden, None, mask, noise=False)
    v_c1 = _policy_value(agent, critic1, critic1, obs, state, hidden, None, mask, noise=False)
    v_c2 = _policy_value(agent, critic2, critic2, obs, state, hidden, None, mask, noise=False)
    assert torch.allclose(v_min, v_c1, atol=1e-6)                      # min ignores the dominated critic2
    assert torch.all(v_c2 > v_c1)                                      # the +100 shift is real
    assert torch.allclose(                                             # symmetric in argument order
        _policy_value(agent, critic2, critic1, obs, state, hidden, None, mask, noise=False), v_c1, atol=1e-6)


def test_pruning_useless_message_leaves_action_unchanged() -> None:
    torch.manual_seed(0)
    agent = _agent()
    j = 1
    with torch.no_grad():
        agent.head.weight[:, MSG_OFFSET + j * 2 : MSG_OFFSET + (j + 1) * 2] = 0.0
    obs, hidden = torch.randn(4, 3, 5), torch.randn(4, 3, 16)
    full = agent.comm_mask(obs, selector_on=False)
    pruned = full.clone()
    pruned[:, :, j] = 0.0

    full_logits, _ = agent(obs, hidden, full)
    pruned_logits, _ = agent(obs, hidden, pruned)
    assert torch.equal(full_logits, pruned_logits)
    assert torch.equal(full_logits.argmax(dim=-1), pruned_logits.argmax(dim=-1))


def test_smv_estimator_is_exact_for_small_coalitions(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    critic1 = _critic()
    critic2 = copy.deepcopy(critic1)                                    # identical twins -> min = critic1
    obs, state, hidden = torch.randn(2, 3, 5), torch.randn(2, 15), torch.randn(2, 3, 16)

    def q_of(i: int, coalition: tuple[int, ...]) -> torch.Tensor:
        mask = torch.zeros(2, 3, 3)
        for sender in coalition:
            mask[:, i, sender] = 1.0
        logits, _ = agent(obs, hidden, mask)
        return critic1(obs, state, torch.softmax(logits, dim=-1))[:, i, 0]

    ascending = torch.tensor([[1, 2], [0, 2], [0, 1]])
    calls: list[int] = []

    def fixed_orderings(probs, num_samples, **kwargs):
        calls.append(num_samples)
        order = ascending if len(calls) % 2 == 1 else ascending.flip(1)
        return order.repeat(2, 1)

    monkeypatch.setattr(torch, "multinomial", fixed_orderings)
    smv = shapley_message_values(agent, critic1, critic2, obs, state, hidden, sample_size=2, noise=False)

    for i in range(3):
        others = [j for j in range(3) if j != i]
        for j in others:
            k = next(o for o in others if o != j)
            exact = 0.5 * (q_of(i, (j,)) - q_of(i, ())) + 0.5 * (q_of(i, (j, k)) - q_of(i, (k,)))
            assert torch.allclose(smv[:, i, j], exact, atol=1e-6)
        assert torch.allclose(smv[:, i, i], q_of(i, ()), atol=1e-6)     # diagonal keeps Q(empty)

    calls.clear()
    single = shapley_message_values(agent, critic1, critic2, obs, state, hidden, sample_size=1, noise=False)
    for i in range(3):
        others = [j for j in range(3) if j != i]
        assert torch.allclose(single[:, i, others].sum(dim=-1), q_of(i, tuple(others)) - q_of(i, ()), atol=1e-5)


def test_selector_gate_activates_after_threshold() -> None:
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 16)
    no_diag = (1.0 - torch.eye(3)).expand(2, 3, 3)

    with torch.no_grad():
        agent.selector[-1].weight.zero_()
        agent.selector[-1].bias.fill_(-1.0)                             # hard-negative scores
    assert torch.equal(agent.comm_mask(obs, selector_on=False), no_diag)

    assert torch.equal(agent.comm_mask(obs, selector_on=True), torch.zeros(2, 3, 3))
    gated_logits, _ = agent(obs, hidden, agent.comm_mask(obs, selector_on=True))
    silent_logits, _ = agent(obs, hidden, torch.zeros(2, 3, 3))
    assert torch.equal(gated_logits, silent_logits)

    with torch.no_grad():
        agent.selector[-1].bias.fill_(1.0)
    assert torch.equal(agent.comm_mask(obs, selector_on=True), no_diag)
    open_logits, _ = agent(obs, hidden, agent.comm_mask(obs, selector_on=True))
    full_logits, _ = agent(obs, hidden, no_diag)
    assert torch.equal(open_logits, full_logits)


def test_selector_regression_ignores_diagonal() -> None:
    torch.manual_seed(0)
    selector = _agent().selector
    inputs = torch.randn(6, 3, INPUT_DIM)
    labels = torch.randn(6, 3, 3)
    poisoned = labels + 1000.0 * torch.eye(3)

    grads = []
    for target in (labels, poisoned):
        loss = train_sms._selector_loss(selector(inputs), target)
        grads.append(torch.autograd.grad(loss, list(selector.parameters())))
    for clean, dirty in zip(*grads):
        assert torch.equal(clean, dirty)

    zero_diag = labels * (1.0 - torch.eye(3))
    loss = train_sms._selector_loss(selector(inputs), zero_diag)
    off = ~torch.eye(3, dtype=torch.bool).expand(6, 3, 3)
    expected = ((selector(inputs)[off] - zero_diag[off]) ** 2).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_recurrence_is_message_independent() -> None:
    torch.manual_seed(0)
    agent = _agent()
    obs_seq = torch.randn(2, 5, 3, 5)
    hidden_a = agent.init_state(2, torch.device("cpu"))
    hidden_b = agent.init_state(2, torch.device("cpu"))
    diverged = False
    for t in range(5):
        mask_a = (torch.rand(2, 3, 3) > 0.5).float()
        mask_b = (torch.rand(2, 3, 3) > 0.5).float()
        logits_a, hidden_a = agent(obs_seq[:, t], hidden_a, mask_a)
        logits_b, hidden_b = agent(obs_seq[:, t], hidden_b, mask_b)
        assert torch.equal(hidden_a, hidden_b)
        diverged = diverged or not torch.equal(logits_a, logits_b)
    assert diverged


def test_smv_labels_scaled_by_mixer_k(monkeypatch) -> None:
    torch.manual_seed(0)
    mixer = LinearDecompositionMixer(3, 15, embed_dim=8)
    state = torch.randn(4, 15)
    k = mixer.k(state)
    assert tuple(k.shape) == (4, 3)
    assert torch.all(k >= 0.0)
    assert torch.allclose(k.sum(dim=1), torch.ones(4), atol=1e-6)
    qs = torch.randn(4, 3)
    assert torch.allclose(mixer(qs, state), (k * qs).sum(dim=1), atol=1e-6)

    learner = _learner()
    learner["mixer"] = mixer
    learner["target_mixer"] = copy.deepcopy(mixer)
    batch = _episode_batch(horizon=4, length=3)

    monkeypatch.setattr(train_sms, "shapley_message_values",
                        lambda *args, **kwargs: torch.ones(args[3].shape[0], 3, 3))
    captured = {}
    real_loss = train_sms._selector_loss

    def spy(preds, labels):
        captured["labels"] = labels.detach().clone()
        return real_loss(preds, labels)

    monkeypatch.setattr(train_sms, "_selector_loss", spy)
    _call_update(learner, batch)

    state_real = batch.obs[:, :-1].reshape(1, 4, -1)[batch.mask.bool()]
    expected = mixer.k(state_real).unsqueeze(-1).expand(-1, 3, 3)
    assert torch.allclose(captured["labels"], expected, atol=1e-6)


def test_actor_gradient_flows_through_critic() -> None:
    torch.manual_seed(0)
    agent = _agent(use_rnn=True)                                       # recurrent head uses the whole trunk
    critic = _critic()
    mixer = LinearDecompositionMixer(3, 15, embed_dim=8)
    obs = torch.randn(4, 3, 5)
    state = obs.reshape(4, -1)
    hidden = agent.init_state(4, torch.device("cpu"))

    logits, _ = agent(obs, hidden, agent.comm_mask(obs, selector_on=False))
    probs = torch.softmax(logits, dim=-1)
    loss = -mixer(critic(obs, state, probs).squeeze(-1), state).mean()
    loss.backward()

    for module in (agent.fc1, agent.gru, agent.msg_encoder, agent.head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in agent.selector.parameters())
    for p in list(critic.parameters()) + list(mixer.parameters()):
        assert p.grad is not None
    actor_ids = {id(p) for p in train_sms._actor_params(agent)}
    for p in list(agent.selector.parameters()) + list(critic.parameters()) + list(mixer.parameters()):
        assert id(p) not in actor_ids


def test_off_policy_update_trains_only_critics_and_mixer() -> None:
    torch.manual_seed(0)
    learner = _learner()
    batch = _episode_batch(horizon=4, length=3, seed=1)

    before = {
        "actor": [p.clone() for p in learner["agent"].parameters()],
        "selector": [p.clone() for p in learner["learner_selector"].parameters()],
        "targets": [p.clone() for p in learner["target_agent"].parameters()],
        "critics": [p.clone() for p in list(learner["critic1"].parameters()) + list(learner["critic2"].parameters())],
        "mixer": [p.clone() for p in learner["mixer"].parameters()],
    }
    _call_update(learner, batch, off=True)

    # Off stream steps only the twin critics + mixer.
    for p, q in zip(learner["agent"].parameters(), before["actor"]):
        assert torch.equal(p, q)
    for p, q in zip(learner["learner_selector"].parameters(), before["selector"]):
        assert torch.equal(p, q)
    for p, q in zip(learner["target_agent"].parameters(), before["targets"]):
        assert torch.equal(p, q)
    assert any(not torch.equal(p, q) for p, q in zip(
        list(learner["critic1"].parameters()) + list(learner["critic2"].parameters()), before["critics"]))
    assert any(not torch.equal(p, q) for p, q in zip(learner["mixer"].parameters(), before["mixer"]))


def test_episode_buffer_sample_latest_returns_newest() -> None:
    buffer = EpisodeReplayBuffer(capacity=3, horizon=2, n_agents=1, obs_dim=1)
    for tag in range(4):
        buffer.add_episode(
            obs=np.full((3, 1, 1), float(tag), dtype=np.float32),
            actions=np.zeros((2, 1), dtype=np.int64),
            rewards=np.full(2, float(tag), dtype=np.float32),
            dones=np.zeros(2, dtype=np.float32),
        )
    batch = buffer.sample_latest(2, torch.device("cpu"))
    assert torch.equal(batch.rewards[:, 0], torch.tensor([2.0, 3.0]))


def test_sms_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "sms.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=6,
        seed=5,
        hidden_dim=16,
        msg_dim=2,
        msg_hidden_dim=8,
        selector_hidden_dim=8,
        critic_hidden_dim=16,
        mixer_embed_dim=8,
        on_buffer_episodes=8,
        off_buffer_episodes=8,
        batch_episodes=2,
        off_batch_episodes=2,               # exercise the off-policy stream in the smoke
        warmup_episodes=2,
        collection_batch_size=2,
        t_selector=10,
        target_update_every=3,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "sms"
    assert len(summary["returns"]) == 6
    assert summary["replay_capacities"] == {"on_policy": 8, "off_policy": 8}
    assert len(summary["initial_evaluation"]["returns"]) == 2
    assert len(summary["random_evaluation"]["returns"]) == 2
    assert len(summary["final_evaluation"]["returns"]) == 2
    assert 0.0 <= summary["communication_rate"] <= 1.0
    assert summary["config"]["t_selector"] == 10
    assert summary["config"]["collection_batch_size"] == 2
    assert checkpoint.exists()
    state_dict = torch.load(str(checkpoint), weights_only=True)
    for key in ("agent", "critic1", "critic2", "mixer", "selector"):
        assert key in state_dict


def test_sms_defaults_match_released_training_config() -> None:
    defaults = {
        name: parameter.default for name, parameter in inspect.signature(train).parameters.items()
    }
    assert defaults["on_buffer_episodes"] == 128
    assert defaults["off_buffer_episodes"] == 5000
    assert defaults["batch_episodes"] == 32
    assert defaults["off_batch_episodes"] == 64
    assert defaults["warmup_episodes"] == 128
    assert defaults["collection_batch_size"] == 8
    assert defaults["updates_per_collection"] == 1
    assert defaults["gamma"] == 0.99
    assert defaults["t_selector"] == 300_000


def test_sms_updates_once_after_each_complete_eight_trajectory_group(monkeypatch) -> None:
    update_modes: list[bool] = []

    def record_update(*args, off=False, **kwargs):
        update_modes.append(off)

    monkeypatch.setattr(train_sms, "_update", record_update)
    train(
        env="navigation",
        n_agents=3,
        horizon=2,
        episodes=17,
        seed=5,
        hidden_dim=16,
        msg_dim=2,
        msg_hidden_dim=8,
        selector_hidden_dim=8,
        critic_hidden_dim=16,
        mixer_embed_dim=8,
        on_buffer_episodes=16,
        off_buffer_episodes=32,
        batch_episodes=4,
        off_batch_episodes=4,
        warmup_episodes=4,
        collection_batch_size=8,
        evaluation_episodes=1,
    )

    assert update_modes == [True, False, True, False]


def test_sms_entropy_matches_release_and_is_finite_at_zero() -> None:
    entropy = torch.tensor([[1.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0]])
    objective = train_sms._normalised_entropy(entropy, mask)
    assert objective.item() == pytest.approx(1.0)
    objective.backward()
    assert torch.allclose(entropy.grad, torch.full_like(entropy, 0.25))
    assert torch.isfinite(train_sms._normalised_entropy(torch.zeros(1, 1), torch.ones(1, 1)))


def test_sms_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = tmp_path / "sms.pt"
    train(env="navigation", n_agents=3, horizon=6, episodes=4, seed=5,
          hidden_dim=16, msg_dim=2, msg_hidden_dim=8, selector_hidden_dim=8,
          critic_hidden_dim=16, mixer_embed_dim=8,
          on_buffer_episodes=8, off_buffer_episodes=8,
          batch_episodes=2, off_batch_episodes=2, warmup_episodes=2,
          collection_batch_size=2,
          t_selector=10, target_update_every=3, checkpoint=str(checkpoint))
    state_dict = torch.load(str(checkpoint), weights_only=True)

    environment = make_env("navigation", 3, 6, 5)
    agent = SMSAgent(n_agents=3, obs_dim=environment.obs_dim, action_dim=environment.num_actions,
                     hidden_dim=16, msg_dim=2, msg_hidden_dim=8, selector_hidden_dim=8)
    critic1 = SMSCritic(3, environment.obs_dim, 3 * environment.obs_dim, environment.num_actions, hidden_dim=16)
    critic2 = SMSCritic(3, environment.obs_dim, 3 * environment.obs_dim, environment.num_actions, hidden_dim=16)
    mixer = LinearDecompositionMixer(3, 3 * environment.obs_dim, embed_dim=8)
    agent.load_state_dict(state_dict["agent"])
    critic1.load_state_dict(state_dict["critic1"])
    critic2.load_state_dict(state_dict["critic2"])
    mixer.load_state_dict(state_dict["mixer"])
    agent.selector.load_state_dict(state_dict["selector"])              # learner copy fits the actor slot

    obs = torch.randn(1, 3, environment.obs_dim)
    logits, _ = agent(obs, agent.init_state(1, torch.device("cpu")), agent.comm_mask(obs, selector_on=True))
    assert torch.isfinite(logits).all()


def test_sms_td_lambda_matches_official_recursion_and_closed_forms() -> None:
    torch.manual_seed(0)
    gamma = 0.9
    rewards = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 0.0]])
    dones = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    values = torch.randn(2, 4)

    def official(rew, term, msk, vals, lam):
        ret = torch.zeros_like(vals)
        ret[:, -1] = vals[:, -1] * (1 - term.sum(dim=1).clamp(max=1.0))
        for t in range(rew.shape[1] - 1, -1, -1):
            ret[:, t] = lam * gamma * ret[:, t + 1] + msk[:, t] * (
                rew[:, t] + (1 - lam) * gamma * vals[:, t + 1] * (1 - term[:, t])
            )
        return ret[:, :-1]

    for lam in (0.0, 0.6, 1.0):
        got = train_sms._td_lambda_targets(rewards, dones, mask, values, gamma, lam)
        assert torch.allclose(got, official(rewards, dones, mask, values, lam), atol=1e-6)

    got0 = train_sms._td_lambda_targets(rewards, dones, mask, values, gamma, 0.0)
    expected_step0 = rewards[0, 0] + gamma * values[0, 1]
    assert got0[0, 0] == pytest.approx(float(expected_step0))
    assert got0[1, 1] == pytest.approx(float(rewards[1, 1]))


def test_sms_targets_invariant_to_padding_after_termination() -> None:
    torch.manual_seed(0)
    gamma, lam = 0.9, 0.6
    rewards = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    dones = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    values_a = torch.randn(1, 5)
    values_b = values_a.clone()
    values_b[:, 2:] += 100.0

    got_a = train_sms._td_lambda_targets(rewards, dones, mask, values_a, gamma, lam)
    got_b = train_sms._td_lambda_targets(rewards, dones, mask, values_b, gamma, lam)
    assert torch.allclose(got_a[:, :2], got_b[:, :2], atol=1e-6)
    assert got_a[0, 1] == pytest.approx(2.0)


def test_sms_smv_orderings_never_contain_the_receiver(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    critic1, critic2 = _critic(), _critic()
    obs = torch.randn(2, 3, 5)
    state = obs.reshape(2, -1)
    hidden = agent.init_state(2, obs.device)

    captured = []
    real_multinomial = torch.multinomial

    def spy(input, num_samples, *args, **kwargs):
        out = real_multinomial(input, num_samples, *args, **kwargs)
        captured.append(out.clone())
        return out

    monkeypatch.setattr(torch, "multinomial", spy)
    shapley_message_values(agent, critic1, critic2, obs, state, hidden, sample_size=2)
    assert captured, "the sampler must draw orderings through torch.multinomial"
    for draw in captured:
        rows = draw.reshape(-1, draw.shape[-1])
        for row in rows:
            assert len(set(row.tolist())) == row.shape[0]


def test_sms_update_leaves_acting_selector_and_targets_untouched() -> None:
    torch.manual_seed(0)
    learner = _learner()
    replay = EpisodeReplayBuffer(capacity=4, horizon=4, n_agents=3, obs_dim=5)
    rng = np.random.default_rng(0)
    for _ in range(3):
        replay.add_episode(
            obs=rng.standard_normal((5, 3, 5)).astype(np.float32),
            actions=rng.integers(0, 4, (4, 3)),
            rewards=rng.standard_normal(4).astype(np.float32),
            dones=np.zeros(4, dtype=np.float32),
        )
    before_selector = [p.clone() for p in learner["agent"].selector.parameters()]
    before_targets = [p.clone() for p in learner["target_agent"].parameters()]

    _call_update(learner, replay.sample_latest(3, torch.device("cpu")), off=False)

    for p, q in zip(learner["agent"].selector.parameters(), before_selector):
        assert torch.equal(p, q)                                        # acting selector untouched (learner copy trains)
    for p, q in zip(learner["target_agent"].parameters(), before_targets):
        assert torch.equal(p, q)


def test_shapley_values_telescope_across_explicit_coalitions() -> None:
    # Each sampled permutation backward-differences nested coalition values, so the
    # marginals must telescope exactly. By construction the diagonal slot holds the
    # receiver's EMPTY-coalition value, so a full row sums to Q_i(full) while the
    # off-diagonal senders sum to Q_i(full) - Q_i(empty). An axis slip, a dropped
    # marginal, or a lost ego-prepended term breaks one of these three identities.
    # `shapley_message_values` allocates in the default dtype, so this runs in float32.
    torch.manual_seed(101)
    agent = _agent()
    critic1, critic2 = _critic(), _critic()
    batch, n = 3, agent.n_agents
    obs = torch.randn(batch, n, 5)
    state = torch.randn(batch, 15)
    hidden = agent.init_state(batch, torch.device("cpu"))

    smv = shapley_message_values(
        agent, critic1, critic2, obs, state, hidden, sample_size=4, noise=False,
    )
    assert smv.shape == (batch, n, n)

    def coalition_value(mask: torch.Tensor) -> torch.Tensor:
        logits, _ = agent(obs, hidden, mask, None, noise=False)
        probs = torch.softmax(logits, dim=-1)
        return torch.min(
            critic1(obs, state, probs), critic2(obs, state, probs),
        ).squeeze(-1)

    grand = coalition_value((1.0 - torch.eye(n)).expand(batch, n, n))
    empty = coalition_value(torch.zeros(batch, n, n))
    diagonal = smv.diagonal(dim1=-2, dim2=-1)

    torch.testing.assert_close(diagonal, empty, rtol=0.0, atol=1e-4)
    torch.testing.assert_close(smv.sum(dim=-1), grand, rtol=0.0, atol=1e-4)
    torch.testing.assert_close(
        smv.sum(dim=-1) - diagonal, grand - empty, rtol=0.0, atol=1e-4,
    )
