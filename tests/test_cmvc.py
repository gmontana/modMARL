"""Golden tests for CMVC's counterfactual requests and monotone aggregator."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from modmarl.algorithms.cmvc import CMVCConfig, CMVCCritic, CMVCLearner, CMVCPolicy
from modmarl.algorithms.cmvc.algorithm import (
    CMVCMonotonicAggregator,
    shift_cmv_labels,
)
from modmarl.common.replay import EpisodeBatch


def _config(**changes) -> CMVCConfig:
    values = {
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 16,
        "hyper_hidden_dim": 8,
        "message_hidden_dim": 6,
        "batch_size": 2,
        "replay_capacity": 8,
        "communication_warmup_updates": 1,
        "target_update_interval": 2,
    }
    values.update(changes)
    return CMVCConfig(**values)


def _batch(batch_size: int = 2, horizon: int = 3) -> EpisodeBatch:
    return EpisodeBatch(
        obs=torch.randn(batch_size, horizon + 1, 3, 5),
        actions=torch.randint(0, 4, (batch_size, horizon, 3)),
        rewards=torch.randn(batch_size, horizon),
        dones=torch.zeros(batch_size, horizon),
        mask=torch.ones(batch_size, horizon),
    )


def test_cmvc_defaults_match_cooperative_navigation_settings() -> None:
    config = CMVCConfig()
    assert config.actor_hidden_dim == 64
    assert config.critic_hidden_dim == 128
    assert config.hyper_hidden_dim == 32
    assert config.gamma == pytest.approx(0.95)
    assert config.actor_learning_rate == pytest.approx(1e-3)
    assert config.critic_learning_rate == pytest.approx(1e-2)
    assert config.replay_capacity == 5_000
    assert config.batch_size == 32
    assert config.pruning_percentile == pytest.approx(70.0)


def test_cmvc_optimizers_keep_the_paper_actor_and_critic_rates_separate() -> None:
    learner = CMVCLearner(3, 5, 4, 3, _config())
    assert learner.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert learner.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)
    assert learner.selector_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_selector_excludes_self_and_requests_only_positive_cmvs() -> None:
    policy = CMVCPolicy(3, 5, 4, _config())
    with torch.no_grad():
        for parameter in policy.message_selector.parameters():
            parameter.zero_()
        policy.message_selector[-1].bias.copy_(torch.tensor([-1.0, 2.0, -3.0]))
    output = policy.step(
        torch.randn(1, 3, 5),
        policy.initial_hidden(1, torch.device("cpu")),
        deterministic=True,
    )
    assert torch.equal(output.cmv_predictions.diagonal(dim1=1, dim2=2), torch.zeros(1, 3))
    assert not output.request_gates.diagonal(dim1=1, dim2=2).any()
    assert torch.equal(output.request_gates[0, :, 1], torch.tensor([True, False, True]))


def test_disabled_communication_has_zero_message_and_no_requests() -> None:
    policy = CMVCPolicy(3, 5, 4, _config())
    output = policy.step(
        torch.randn(1, 3, 5),
        policy.initial_hidden(1, torch.device("cpu")),
        communication_enabled=False,
        deterministic=True,
    )
    assert not output.request_gates.any()
    assert torch.equal(output.messages, torch.zeros_like(output.messages))


def test_monotone_aggregator_has_nonnegative_input_derivative() -> None:
    torch.manual_seed(0)
    aggregator = CMVCMonotonicAggregator(3, 2, 4, 5, 6)
    embeddings = torch.rand(2, 3, 2, requires_grad=True)
    values = torch.ones(2, 3, 3)
    values.diagonal(dim1=1, dim2=2).zero_()
    messages, _ = aggregator(embeddings, values)
    messages.sum().backward()
    assert embeddings.grad is not None
    assert torch.all(embeddings.grad >= 0)


def test_counterfactual_value_is_full_q_minus_zero_sender_action() -> None:
    critic = CMVCCritic(3, 2, 2, hidden_dim=4)
    joint_dim = 3 * (2 + 2)
    linear = nn.Linear(joint_dim, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    critic.net = linear
    obs = torch.zeros(1, 3, 2)
    actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    values = critic.counterfactual_values(obs, actions)
    expected = torch.tensor([[[0.0, 7.0, 11.0], [3.0, 0.0, 11.0], [3.0, 7.0, 0.0]]])
    assert torch.equal(values, expected)


def test_recurrence_changes_trajectory_embedding_for_same_observation() -> None:
    torch.manual_seed(1)
    policy = CMVCPolicy(3, 5, 4, _config())
    obs = torch.randn(1, 3, 5)
    zero = policy.initial_hidden(1, torch.device("cpu"))
    first = policy.step(obs, zero, deterministic=True)
    second = policy.step(obs, first.hidden, deterministic=True)
    assert not torch.allclose(first.trajectory_embeddings, second.trajectory_embeddings)


def test_percentile_threshold_shifts_labels_and_keeps_self_zero() -> None:
    values = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]],
            [[0.0, 7.0, 8.0], [9.0, 0.0, 10.0], [11.0, 12.0, 0.0]],
        ],
    )
    shifted = shift_cmv_labels(values, torch.tensor([1.0, 0.0]), percentile=50.0)
    # The first valid row's off-diagonal median is 3.5.
    assert shifted[0, 0, 1] == pytest.approx(-2.5)
    assert shifted[0, 2, 1] == pytest.approx(2.5)
    assert torch.equal(shifted.diagonal(dim1=1, dim2=2), torch.zeros(2, 3))


def test_actor_path_does_not_backpropagate_into_selector() -> None:
    torch.manual_seed(2)
    policy = CMVCPolicy(3, 5, 4, _config())
    output = policy.unroll(torch.randn(2, 3, 3, 5), hard=False)
    output.action_vectors.sum().backward()
    assert all(parameter.grad is None for parameter in policy.message_selector.parameters())
    assert any(parameter.grad is not None for parameter in policy.message_aggregator.parameters())


def test_learner_warmup_and_hard_target_cadence() -> None:
    torch.manual_seed(3)
    learner = CMVCLearner(3, 5, 4, 3, _config())
    assert not learner.communication_enabled
    initial_target = copy.deepcopy(learner.target_policy.state_dict())
    first = learner.update(_batch())
    assert first is not None
    assert learner.communication_enabled
    assert all(
        torch.equal(initial_target[name], learner.target_policy.state_dict()[name])
        for name in initial_target
    )
    second = learner.update(_batch())
    assert second is not None
    assert all(
        torch.equal(learner.policy.state_dict()[name], learner.target_policy.state_dict()[name])
        for name in learner.policy.state_dict()
    )


def test_padding_does_not_contribute_to_losses_or_communication_rate() -> None:
    torch.manual_seed(4)
    learner = CMVCLearner(3, 5, 4, 3, _config(communication_warmup_updates=0))
    batch = _batch(batch_size=1)
    batch.mask[:, -1] = 0.0
    batch.rewards[:, -1] = 1e6
    result = learner.update(batch)
    assert result is not None
    assert torch.isfinite(torch.tensor(result.critic_loss))
    assert 0.0 <= result.communication_rate <= 1.0


def test_cmvc_trainer_smoke_and_checkpoint(tmp_path) -> None:
    from examples.train_cmvc import train

    checkpoint = tmp_path / "cmvc.pt"
    summary = train(
        episodes=2,
        evaluation_episodes=1,
        updates_per_episode=1,
        config=_config(communication_warmup_updates=0),
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "cmvc"
    assert "GaoZiHong/CMVC-empty" in summary["source_revision"]
    assert len(summary["returns"]) == 2
    assert 0.0 <= summary["communication_rate"] <= 1.0
    assert checkpoint.exists()
