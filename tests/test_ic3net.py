"""Golden tests for IC3Net communication and individualized REINFORCE."""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_ic3net import (
    _agent_alive,
    _agent_rewards,
    _collect_episode,
    _pad_episodes,
    train,
)
from marl_envs import make_env
from modmarl.algorithms.ic3net import IC3NetAgent, IC3NetCell


def test_cell_shapes() -> None:
    cell = IC3NetCell(5, 4, 16)
    outputs = cell(
        torch.randn(6, 3, 5),
        torch.randn(6, 3, 16),
        torch.randn(6, 3, 16),
        torch.randint(0, 2, (6, 3)),
    )
    assert [tensor.shape for tensor in outputs] == [
        (6, 3, 4),
        (6, 3, 2),
        (6, 3),
        (6, 3, 16),
        (6, 3, 16),
    ]


def test_closed_gates_remove_sender_hidden_content() -> None:
    torch.manual_seed(0)
    cell = IC3NetCell(5, 4, 16)
    silenced = copy.deepcopy(cell)
    with torch.no_grad():
        silenced.comm.weight.zero_()
    obs = torch.randn(4, 3, 5)
    hidden = torch.randn(4, 3, 16)
    state = torch.randn(4, 3, 16)
    gates = torch.zeros(4, 3, dtype=torch.long)
    for actual, expected in zip(cell(obs, hidden, state, gates), silenced(obs, hidden, state, gates)):
        torch.testing.assert_close(actual, expected)


def test_silent_sender_leaks_nothing_but_still_listens() -> None:
    torch.manual_seed(0)
    cell = IC3NetCell(5, 4, 16)
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 16)
    state = torch.randn(2, 3, 16)
    gates = torch.tensor([[1, 1, 0], [1, 1, 0]])
    base_logits, _, base_value, _, _ = cell(obs, hidden, state, gates)
    changed_hidden = hidden.clone()
    changed_hidden[:, 2] += 1.0
    changed_logits, _, changed_value, _, _ = cell(obs, changed_hidden, state, gates)
    torch.testing.assert_close(base_logits[:, :2], changed_logits[:, :2])
    torch.testing.assert_close(base_value[:, :2], changed_value[:, :2])
    assert not torch.allclose(base_logits[:, 2], changed_logits[:, 2])


def test_open_sender_reaches_other_agents() -> None:
    torch.manual_seed(0)
    cell = IC3NetCell(5, 4, 16)
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 16)
    state = torch.randn(2, 3, 16)
    gates = torch.ones(2, 3, dtype=torch.long)
    base, *_ = cell(obs, hidden, state, gates)
    hidden[:, 2] += 1.0
    changed, *_ = cell(obs, hidden, state, gates)
    assert not torch.allclose(base[:, 0], changed[:, 0])
    assert not torch.allclose(base[:, 1], changed[:, 1])


def test_single_agent_has_finite_zero_peer_communication() -> None:
    cell = IC3NetCell(5, 4, 16)
    outputs = cell(
        torch.randn(4, 1, 5),
        torch.randn(4, 1, 16),
        torch.randn(4, 1, 16),
        torch.ones(4, 1, dtype=torch.long),
    )
    assert all(torch.isfinite(tensor).all() for tensor in outputs)


def test_recurrent_replay_reproduces_joint_log_probabilities() -> None:
    torch.manual_seed(0)
    agent = IC3NetAgent(5, 4, 16)
    observations = torch.randn(4, 1, 3, 5)
    hidden, cell, previous_gate = agent.init_state(1, 3, observations.device)
    stored = []
    with torch.no_grad():
        for step in range(4):
            action, action_log, gate, gate_log, _, hidden, cell = agent.act(
                observations[step], hidden, cell, previous_gate,
            )
            stored.append((action, gate, action_log + gate_log))
            previous_gate = gate
    hidden, cell, previous_gate = agent.init_state(1, 3, observations.device)
    with torch.no_grad():
        for step, (action, gate, expected) in enumerate(stored):
            actual, _, _, hidden, cell = agent.evaluate_step(
                observations[step], hidden, cell, previous_gate, action, gate,
            )
            torch.testing.assert_close(actual, expected)
            previous_gate = gate


def test_individual_rewards_produce_individual_returns() -> None:
    torch.manual_seed(0)
    agent = IC3NetAgent(2, 2, 8, learning_rate=0.0, gamma=1.0)
    observations = torch.randn(1, 2, 2, 2)
    actions = torch.zeros(1, 2, 2, dtype=torch.long)
    gates = torch.zeros_like(actions)
    rewards = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]])
    update = agent.update(observations, actions, gates, rewards, torch.ones_like(rewards))
    assert update.return_mean == pytest.approx((3.0 + 30.0 + 2.0 + 20.0) / 4)


def test_individual_continuation_stops_only_completed_agents_return() -> None:
    agent = IC3NetAgent(2, 2, 8, learning_rate=0.0)
    update = agent.update(
        torch.randn(1, 2, 2, 2),
        torch.zeros(1, 2, 2, dtype=torch.long),
        torch.zeros(1, 2, 2, dtype=torch.long),
        torch.tensor([[[1.0, 10.0], [2.0, 20.0]]]),
        torch.ones(1, 2, 2),
        continuation=torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
    )
    assert update.return_mean == pytest.approx((1.0 + 30.0 + 2.0 + 20.0) / 4)


def test_gate_head_receives_reinforce_gradient() -> None:
    torch.manual_seed(1)
    agent = IC3NetAgent(2, 2, 8, learning_rate=0.0)
    agent.update(
        torch.randn(2, 3, 2, 2),
        torch.randint(0, 2, (2, 3, 2)),
        torch.randint(0, 2, (2, 3, 2)),
        torch.randn(2, 3, 2),
        torch.ones(2, 3, 2),
    )
    gradient = agent.cell.gate_head.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_optimizer_and_objective_defaults_match_release() -> None:
    agent = IC3NetAgent(2, 2)
    group = agent.optimizer.param_groups[0]
    assert isinstance(agent.optimizer, torch.optim.RMSprop)
    assert group["lr"] == pytest.approx(1e-3)
    assert group["alpha"] == pytest.approx(0.97)
    assert group["eps"] == pytest.approx(1e-6)
    assert agent.gamma == 1.0
    assert agent.value_coefficient == 0.01
    assert agent.entropy_coefficient == 0.0
    assert agent.detach_gap == 10
    assert agent.cell.hidden_dim == 128


def test_reward_adapter_preserves_vectors_and_expands_shared_scalars() -> None:
    assert _agent_rewards(2.0, {}, 3).tolist() == [2.0, 2.0, 2.0]
    assert _agent_rewards(0.0, {"agent_rewards": [1.0, 2.0, 3.0]}, 3).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="shape"):
        _agent_rewards(0.0, {"agent_rewards": [1.0, 2.0]}, 3)


def test_alive_adapter_preserves_individual_activity() -> None:
    assert _agent_alive({}, 3).tolist() == [1.0, 1.0, 1.0]
    assert _agent_alive({"alive_mask": [1.0, 0.0, 1.0]}, 3).tolist() == [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="shape"):
        _agent_alive({"alive_mask": [1.0, 0.0]}, 3)


def test_episode_collection_and_padding_keep_agent_reward_axis() -> None:
    environment = make_env("navigation", 2, 4, 5)
    agent = IC3NetAgent(environment.obs_dim, environment.num_actions, 16)
    episode, _ = _collect_episode(agent, environment, 2, 5, torch.device("cpu"))
    batch = _pad_episodes([episode], 2, environment.obs_dim, torch.device("cpu"))
    assert batch["rewards"].shape == (1, 4, 2)
    assert batch["mask"].shape == (1, 4, 2)
    assert batch["alive"].shape == (1, 4, 2)
    assert batch["continuation"][0, -1].sum() == 0


def test_permutation_equivariance() -> None:
    torch.manual_seed(0)
    cell = IC3NetCell(5, 4, 16)
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 16)
    state = torch.randn(2, 3, 16)
    gate = torch.tensor([[1, 0, 1], [0, 1, 1]])
    permutation = torch.tensor([2, 0, 1])
    original = cell(obs, hidden, state, gate)
    permuted = cell(obs[:, permutation], hidden[:, permutation], state[:, permutation], gate[:, permutation])
    for actual, expected in zip(permuted, original):
        torch.testing.assert_close(actual, expected[:, permutation])


def test_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "ic3net.pt"
    summary = train(
        env="navigation",
        n_agents=2,
        horizon=4,
        episodes=3,
        seed=5,
        hidden_dim=16,
        batch_steps=4,
        evaluation_episodes=1,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "ic3net"
    assert summary["episodes"] == 3
    assert checkpoint.exists()
    checkpoint_state = torch.load(checkpoint, weights_only=True)
    assert any(key.startswith("cell") for key in checkpoint_state["agent"])
