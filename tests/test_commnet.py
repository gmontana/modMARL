"""Golden tests for released recurrent CommNet and cooperative REINFORCE."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_commnet import _collect_episode, _pad_episodes, train
from marl_envs import make_env
from modmarl import CommNetActor, CommNetAgent
from modmarl.algorithms.commnet import CommNetCell


def test_lstm_cell_matches_released_gate_equation() -> None:
    torch.manual_seed(0)
    cell = CommNetCell(5, 7)
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 7)
    memory = torch.randn(2, 3, 7)
    received = torch.randn(2, 3, 7)
    alive = torch.ones(2, 3)
    actual_hidden, actual_memory, actual_received = cell(
        obs, hidden, memory, received, alive,
    )
    preactivation = cell.obs_encoder(obs) + cell.hidden_encoder(hidden) + cell.comm_encoder(received)
    forget, write, read, candidate = preactivation.chunk(4, dim=-1)
    expected_memory = memory * forget.sigmoid() + candidate.tanh() * write.sigmoid()
    expected_hidden = expected_memory.tanh() * read.sigmoid()
    torch.testing.assert_close(actual_memory, expected_memory)
    torch.testing.assert_close(actual_hidden, expected_hidden)
    torch.testing.assert_close(actual_received, CommNetCell.receive(expected_hidden, alive))


def test_received_message_is_other_active_agent_mean() -> None:
    hidden = torch.tensor([[[1.0], [3.0], [8.0]]])
    alive = torch.tensor([[1.0, 1.0, 0.0]])
    received = CommNetCell.receive(hidden, alive)
    assert received.squeeze(-1).tolist() == [[3.0, 1.0, 0.0]]
    single = CommNetCell.receive(torch.tensor([[[4.0]]]), torch.ones(1, 1))
    torch.testing.assert_close(single, torch.zeros_like(single))


def test_initial_hidden_and_cell_match_release_and_message_is_zero() -> None:
    actor = CommNetActor(5, 4, 16)
    hidden, cell, received = actor.initial_state(2, 3, torch.device("cpu"))
    torch.testing.assert_close(hidden, torch.full((2, 3, 16), 0.1))
    torch.testing.assert_close(cell, torch.full((2, 3, 16), 0.1))
    torch.testing.assert_close(received, torch.zeros(2, 3, 16))


def test_message_affects_other_agents_on_next_environment_step() -> None:
    torch.manual_seed(0)
    actor = CommNetActor(5, 4, 16)
    obs = torch.randn(2, 3, 5)
    changed = obs.clone()
    changed[:, 2] += 1.0
    state = actor.initial_state(2, 3, obs.device)
    logits, _, hidden, cell, received = actor.step(obs, *state, torch.ones(2, 3))
    changed_logits, _, changed_hidden, changed_cell, changed_received = actor.step(
        changed, *state, torch.ones(2, 3),
    )
    torch.testing.assert_close(logits[:, :2], changed_logits[:, :2])
    next_obs = torch.randn_like(obs)
    next_logits, *_ = actor.step(
        next_obs, hidden, cell, received, torch.ones(2, 3),
    )
    changed_next_logits, *_ = actor.step(
        next_obs, changed_hidden, changed_cell, changed_received, torch.ones(2, 3),
    )
    assert not torch.allclose(next_logits[:, 0], changed_next_logits[:, 0])
    assert not torch.allclose(next_logits[:, 1], changed_next_logits[:, 1])


def test_policy_is_agent_permutation_equivariant() -> None:
    torch.manual_seed(0)
    actor = CommNetActor(5, 4, 16)
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 16)
    cell = torch.randn(2, 3, 16)
    received = torch.randn(2, 3, 16)
    alive = torch.tensor([[1.0, 0, 1], [1, 1, 1]])
    permutation = torch.tensor([2, 0, 1])
    original = actor.step(obs, hidden, cell, received, alive)
    permuted = actor.step(
        obs[:, permutation],
        hidden[:, permutation],
        cell[:, permutation],
        received[:, permutation],
        alive[:, permutation],
    )
    for actual, expected in zip(permuted, original):
        torch.testing.assert_close(actual, expected[:, permutation])


def test_update_matches_undiscounted_reinforce_and_baseline_losses() -> None:
    torch.manual_seed(0)
    agent = CommNetAgent(3, 2, 8, learning_rate=0.0)
    obs = torch.randn(1, 2, 2, 3)
    actions = torch.tensor([[[0, 1], [1, 0]]])
    rewards = torch.tensor([[1.0, 2.0]])
    mask = torch.ones(1, 2, 2)
    hidden, cell, received = agent.actor.initial_state(1, 2, obs.device)
    logits, baselines = [], []
    for step in range(2):
        step_logits, baseline, hidden, cell, received = agent.actor.step(
            obs[:, step], hidden, cell, received, mask[:, step],
        )
        logits.append(step_logits)
        baselines.append(baseline)
    logits = torch.stack(logits, dim=1)
    baselines = torch.stack(baselines, dim=1)
    returns = torch.tensor([[[3.0, 3.0], [2.0, 2.0]]])
    log_prob = torch.log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    expected_policy = -(log_prob * (returns - baselines.detach())).sum()
    expected_baseline = (baselines - returns).square().sum()
    update = agent.update(obs, actions, rewards, mask)
    assert update.policy_loss == pytest.approx(float(expected_policy.detach()), rel=1e-6)
    assert update.baseline_loss == pytest.approx(float(expected_baseline.detach()), rel=1e-6)
    assert update.return_mean == pytest.approx(2.5)


def test_agent_specific_rewards_are_averaged_for_cooperation() -> None:
    agent = CommNetAgent(3, 2, 8, learning_rate=0.0)
    update = agent.update(
        torch.randn(1, 2, 2, 3),
        torch.zeros(1, 2, 2, dtype=torch.long),
        torch.tensor([[[1.0, 3.0], [2.0, 6.0]]]),
        torch.ones(1, 2, 2),
    )
    assert update.return_mean == pytest.approx((6.0 + 6.0 + 4.0 + 4.0) / 4)


def test_delayed_message_path_receives_policy_gradient() -> None:
    torch.manual_seed(1)
    agent = CommNetAgent(3, 2, 8, learning_rate=0.0)
    agent.update(
        torch.randn(2, 3, 3, 3),
        torch.randint(0, 2, (2, 3, 3)),
        torch.randn(2, 3),
        torch.ones(2, 3, 3),
    )
    gradient = agent.actor.cell.comm_encoder.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_padding_does_not_contribute_to_return() -> None:
    agent = CommNetAgent(3, 2, 8, learning_rate=0.0)
    update = agent.update(
        torch.randn(1, 3, 2, 3),
        torch.zeros(1, 3, 2, dtype=torch.long),
        torch.tensor([[1.0, 2.0, 1000.0]]),
        torch.tensor([[[1.0, 1], [1, 1], [0, 0]]]),
    )
    assert update.return_mean == pytest.approx(2.5)


def test_optimizer_and_unroll_defaults_match_release() -> None:
    agent = CommNetAgent(3, 2)
    group = agent.optimizer.param_groups[0]
    assert isinstance(agent.optimizer, torch.optim.RMSprop)
    assert group["lr"] == pytest.approx(1e-3)
    assert group["alpha"] == pytest.approx(0.97)
    assert group["eps"] == pytest.approx(1e-8)
    assert agent.baseline_coefficient == pytest.approx(0.03)
    assert agent.unroll_length == 10
    assert agent.unroll_frequency == 4
    assert not hasattr(agent, "critic")


def test_collection_and_padding_keep_recurrent_rollout_contract() -> None:
    environment = make_env("navigation", 2, 4, 5)
    agent = CommNetAgent(environment.obs_dim, environment.num_actions, 16)
    episode, _ = _collect_episode(agent, environment, 2, 5, torch.device("cpu"))
    batch = _pad_episodes([episode], 2, environment.obs_dim, torch.device("cpu"))
    assert batch["obs"].shape == (1, 4, 2, environment.obs_dim)
    assert batch["mask"].shape == (1, 4, 2)


def test_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "commnet.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        hidden_dim=16,
        batch_size=2,
        evaluation_episodes=1,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "commnet"
    assert summary["config"]["model"] == "lstm"
    assert summary["communication_rate"] == 1.0
    assert checkpoint.exists()
