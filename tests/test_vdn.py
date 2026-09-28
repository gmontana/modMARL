"""Paper-equation and integration tests for recurrent VDN."""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_vdn import train
from modmarl.algorithms.vdn import VDNAgent, VDNDuelingLSTM
from modmarl.common.replay import EpisodeBatch


def _batch(batch_size: int = 2, time_steps: int = 4, n_agents: int = 3) -> EpisodeBatch:
    return EpisodeBatch(
        obs=torch.randn(batch_size, time_steps + 1, n_agents, 5),
        actions=torch.randint(0, 3, (batch_size, time_steps, n_agents)),
        rewards=torch.randn(batch_size, time_steps),
        dones=torch.zeros(batch_size, time_steps),
        mask=torch.ones(batch_size, time_steps),
    )


def test_dueling_head_is_value_plus_centered_advantage() -> None:
    network = VDNDuelingLSTM(5, 3, hidden_dim=8)
    inputs = torch.randn(2, 4, 5)
    encoded = network.encoder(inputs)
    recurrent, _ = network.lstm(encoded)
    expected = network.value(recurrent) + network.advantage(recurrent)
    expected = expected - network.advantage(recurrent).mean(dim=-1, keepdim=True)
    actual, _ = network(inputs)
    torch.testing.assert_close(actual, expected)


def test_role_information_is_one_hot_and_agent_specific() -> None:
    learner = VDNAgent(3, 5, 3, hidden_dim=8)
    augmented = learner._append_roles(torch.zeros(2, 4, 3, 5))
    assert augmented.shape == (2, 4, 3, 8)
    for agent_index in range(3):
        torch.testing.assert_close(
            augmented[0, 0, agent_index, 5:], torch.eye(3)[agent_index]
        )


def test_lambda_return_matches_forward_view_trace_equation() -> None:
    learner = VDNAgent(2, 5, 3, hidden_dim=8, gamma=0.5, trace_lambda=0.5)
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    dones = torch.zeros_like(rewards)
    mask = torch.ones_like(rewards)
    next_values = torch.tensor([[10.0, 20.0, 30.0]])
    targets = learner._lambda_targets(rewards, dones, mask, next_values)
    torch.testing.assert_close(targets, torch.tensor([[6.375, 11.5, 18.0]]))


def test_lambda_return_stops_at_terminal_and_ignores_padding() -> None:
    learner = VDNAgent(2, 5, 3, hidden_dim=8, gamma=0.5, trace_lambda=0.5)
    rewards = torch.tensor([[1.0, 2.0, 9.0]])
    dones = torch.tensor([[0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    values = torch.tensor([[10.0, 20.0, 30.0]])
    targets = learner._lambda_targets(rewards, dones, mask, values)
    torch.testing.assert_close(targets, torch.tensor([[4.0, 2.0, 0.0]]))


def test_truncated_episode_bootstraps_fully_before_padding() -> None:
    learner = VDNAgent(2, 5, 3, hidden_dim=8, gamma=0.5, trace_lambda=0.5)
    rewards = torch.tensor([[1.0, 2.0, 0.0]])
    dones = torch.zeros_like(rewards)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    next_values = torch.tensor([[10.0, 20.0, 999.0]])

    targets = learner._lambda_targets(rewards, dones, mask, next_values)

    # The final valid transition is a time-limit truncation, so V(s_2)=20 is
    # the complete endpoint bootstrap rather than being attenuated by lambda.
    torch.testing.assert_close(targets, torch.tensor([[6.5, 12.0, 0.0]]))


def test_additive_q_total_gathers_each_agents_selected_utility() -> None:
    torch.manual_seed(0)
    learner = VDNAgent(3, 5, 3, hidden_dim=8, learning_rate=0.0)
    batch = _batch()
    with torch.no_grad():
        all_q = learner._sequence_q(learner.q_network, batch.obs[:, :-1], grad=False)
        expected = all_q.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    metrics = learner.update(batch)
    torch.testing.assert_close(metrics["q_total"], expected)


def test_trace_boundary_detaches_recurrent_gradient() -> None:
    learner = VDNAgent(2, 5, 3, hidden_dim=8, trace_length=2)
    obs = torch.randn(1, 4, 2, 5, requires_grad=True)
    q_values = learner._sequence_q(learner.q_network, obs, grad=True)
    q_values[:, 2].sum().backward()
    assert obs.grad[:, 2].abs().sum() > 0
    assert obs.grad[:, :2].abs().sum() == 0


def test_execution_recurrence_advances_and_resets() -> None:
    learner = VDNAgent(2, 5, 3, hidden_dim=8)
    learner.act(torch.zeros(2, 5), epsilon=0.0)
    first_hidden = tuple(value.clone() for value in learner._execution_hidden)
    learner.act(torch.ones(2, 5), epsilon=0.0)
    assert any(not torch.equal(a, b) for a, b in zip(first_hidden, learner._execution_hidden))
    learner.reset_hidden()
    assert learner._execution_hidden is None


def test_update_changes_online_network_but_not_hard_target() -> None:
    learner = VDNAgent(3, 5, 3, hidden_dim=8, learning_rate=1e-2)
    online_before = copy.deepcopy(learner.q_network.state_dict())
    target_before = copy.deepcopy(learner.target_q_network.state_dict())
    learner.update(_batch())
    assert any(not torch.equal(value, online_before[key]) for key, value in learner.q_network.state_dict().items())
    assert all(torch.equal(value, target_before[key]) for key, value in learner.target_q_network.state_dict().items())
    assert learner.maybe_update_targets(199) is False
    assert learner.maybe_update_targets(200) is True
    for online, target in zip(learner.q_network.parameters(), learner.target_q_network.parameters()):
        torch.testing.assert_close(online, target)


def test_vdn_train_smoke_uses_episode_replay_and_complete_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "vdn.pt"
    summary = train(
        env="navigation", n_agents=3, horizon=6, episodes=5, seed=5,
        hidden_dim=8, buffer_size=16, batch_size=2, warmup_episodes=2,
        target_update_interval=2, evaluation_episodes=2, checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "vdn"
    saved = torch.load(checkpoint, weights_only=True)
    assert "model" in saved and "optimizer" in saved
    assert "q_network.lstm.weight_ih_l0" in saved["model"]
    assert "target_q_network.advantage.2.bias" in saved["model"]
    assert len(summary["final_evaluation"]["returns"]) == 2
