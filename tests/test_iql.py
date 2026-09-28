"""Equation and integration tests for Tampuu et al.'s independent DQNs."""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_iql import train
from modmarl.algorithms import iql
from modmarl.algorithms.iql import IQLAgent, IQLUpdate
from modmarl.common.replay import ReplayBatch


def _batch(*, reward: float = 0.0, done: float = 0.0, batch_size: int = 4) -> ReplayBatch:
    return ReplayBatch(
        obs=torch.zeros(batch_size, 1, 3),
        actions=torch.zeros(batch_size, 1, dtype=torch.int64),
        rewards=torch.full((batch_size,), reward),
        next_obs=torch.ones(batch_size, 1, 3),
        dones=torch.full((batch_size,), done),
    )


def test_each_agent_has_disjoint_online_target_and_optimizer_state() -> None:
    learner = IQLAgent(3, 3, 2, hidden_dim=8)
    online_ids = [{id(p) for p in net.parameters()} for net in learner.q_networks]
    target_ids = [{id(p) for p in net.parameters()} for net in learner.target_q_networks]

    assert all(online_ids[i].isdisjoint(online_ids[j]) for i in range(3) for j in range(i))
    assert all(online_ids[i].isdisjoint(target_ids[i]) for i in range(3))
    assert all(not parameter.requires_grad for parameter in learner.target_q_networks.parameters())
    for index, optimizer in enumerate(learner.optimizers):
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert optimized == online_ids[index]
        assert optimizer.defaults["alpha"] == pytest.approx(0.95)
        assert optimizer.defaults["eps"] == pytest.approx(0.01)


def test_update_returns_structured_diagnostics() -> None:
    learner = IQLAgent(1, 3, 2, hidden_dim=8, learning_rate=0.0)
    update = learner.update_agent(0, _batch(reward=1.0))
    assert isinstance(update, IQLUpdate)
    assert update.loss >= 0.0
    assert update.mean_absolute_td_error >= 0.0


def test_epsilon_uses_paper_step_schedule_after_replay_warmup() -> None:
    learner = IQLAgent(2, 3, 2, learn_start=100, epsilon_anneal_steps=200)
    assert learner.epsilon(0) == pytest.approx(1.0)
    assert learner.epsilon(100) == pytest.approx(1.0)
    assert learner.epsilon(200) == pytest.approx(0.525)
    assert learner.epsilon(300) == pytest.approx(0.05)
    assert learner.epsilon(10_000) == pytest.approx(0.05)


def test_zero_epsilon_selects_each_agents_own_greedy_action() -> None:
    learner = IQLAgent(2, 3, 2, hidden_dim=8)
    with torch.no_grad():
        for parameter in learner.q_networks[0].parameters():
            parameter.zero_()
        for parameter in learner.q_networks[1].parameters():
            parameter.zero_()
        learner.q_networks[0].net[-1].bias.copy_(torch.tensor([2.0, 1.0]))
        learner.q_networks[1].net[-1].bias.copy_(torch.tensor([1.0, 2.0]))

    assert learner.act(torch.zeros(2, 3), epsilon=0.0).tolist() == [0, 1]


def test_td_target_is_reward_clipped_single_agent_dqn_target(monkeypatch) -> None:
    learner = IQLAgent(1, 3, 2, hidden_dim=8, gamma=0.9, reward_clip=1.0)
    with torch.no_grad():
        for parameter in learner.target_q_networks[0].parameters():
            parameter.zero_()
        learner.target_q_networks[0].net[-1].bias.copy_(torch.tensor([2.0, 3.0]))
    captured: dict[str, torch.Tensor] = {}
    original_loss = iql.F.smooth_l1_loss

    def spy(prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        captured["target"] = target.detach().clone()
        return original_loss(prediction, target, **kwargs)

    monkeypatch.setattr(iql.F, "smooth_l1_loss", spy)
    learner.update_agent(0, _batch(reward=7.0))
    torch.testing.assert_close(captured["target"], torch.full((4,), 1.0 + 0.9 * 3.0))


def test_terminal_transition_does_not_bootstrap(monkeypatch) -> None:
    learner = IQLAgent(1, 3, 2, hidden_dim=8, gamma=0.9)
    captured: dict[str, torch.Tensor] = {}
    original_loss = iql.F.smooth_l1_loss

    def spy(prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        captured["target"] = target.detach().clone()
        return original_loss(prediction, target, **kwargs)

    monkeypatch.setattr(iql.F, "smooth_l1_loss", spy)
    learner.update_agent(0, _batch(reward=-3.0, done=1.0))
    torch.testing.assert_close(captured["target"], torch.full((4,), -1.0))


def test_updating_one_agent_cannot_change_another_agent() -> None:
    learner = IQLAgent(2, 3, 2, hidden_dim=8, learning_rate=1e-2)
    before_first = copy.deepcopy(learner.q_networks[0].state_dict())
    before_second = copy.deepcopy(learner.q_networks[1].state_dict())

    learner.update_agent(0, _batch(reward=1.0))

    assert any(not torch.equal(value, before_first[key]) for key, value in learner.q_networks[0].state_dict().items())
    assert all(torch.equal(value, before_second[key]) for key, value in learner.q_networks[1].state_dict().items())


def test_optimizer_matches_released_centered_rmsprop_equation() -> None:
    learner = IQLAgent(1, 3, 2, hidden_dim=8, learning_rate=0.25)
    parameter = next(learner.q_networks[0].parameters())
    with torch.no_grad():
        parameter.fill_(1.0)
    parameter.grad = torch.full_like(parameter, 2.0)

    learner.optimizers[0].step()

    mean_gradient = 0.05 * 2.0
    mean_square = 0.05 * 2.0**2
    denominator = (mean_square - mean_gradient**2 + 0.01) ** 0.5
    expected = torch.full_like(parameter, 1.0 - 0.25 * 2.0 / denominator)
    torch.testing.assert_close(parameter, expected)


def test_targets_are_hard_copied_only_at_configured_environment_step() -> None:
    learner = IQLAgent(2, 3, 2, hidden_dim=8, target_update_interval=5)
    assert learner.maybe_update_targets(1) is True
    target_before = copy.deepcopy(learner.target_q_networks.state_dict())
    with torch.no_grad():
        next(learner.q_networks[0].parameters()).add_(1.0)

    assert learner.maybe_update_targets(5) is False
    assert all(torch.equal(value, target_before[key]) for key, value in learner.target_q_networks.state_dict().items())
    assert learner.maybe_update_targets(6) is True
    for online, target in zip(learner.q_networks, learner.target_q_networks):
        for online_parameter, target_parameter in zip(online.parameters(), target.parameters()):
            torch.testing.assert_close(target_parameter, online_parameter)


def test_iql_train_smoke_and_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = tmp_path / "iql.pt"
    summary = train(
        env="navigation", n_agents=3, horizon=6, episodes=3, seed=5,
        hidden_dim=16, buffer_size=64, batch_size=4, learn_start=0,
        update_every=1, target_update_interval=4, epsilon_anneal_steps=10,
        evaluation_episodes=3, checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "iql"
    state = torch.load(checkpoint, weights_only=True)
    assert "q_networks.0.net.0.weight" in state["model"]
    assert "target_q_networks.2.net.4.bias" in state["model"]
    assert len(state["optimizers"]) == 3
    assert state["total_steps"] == 18
    assert len(summary["initial_evaluation"]["returns"]) == 3
    assert len(summary["random_evaluation"]["successes"]) == 3
    assert len(summary["final_evaluation"]["mean_distances"]) == 3


def test_iql_evaluation_is_reproducible_and_held_out_from_training_rng() -> None:
    kwargs = dict(
        env="navigation", n_agents=2, horizon=4, episodes=0, seed=11,
        hidden_dim=8, buffer_size=16, batch_size=2, learn_start=0,
        evaluation_episodes=4,
    )
    first = train(**kwargs)
    second = train(**kwargs)
    assert first["evaluation_seeds"] == [100_011, 100_012, 100_013, 100_014]
    assert first["initial_evaluation"] == second["initial_evaluation"]
    assert first["random_evaluation"] == second["random_evaluation"]
    assert first["final_evaluation"] == second["final_evaluation"]
