"""Golden tests for the paper-faithful ATOC mechanism and learner."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from marl_envs.noisy_navigation import NoisyNavigationEnv, SignedNoisyNavigationEnv
from modmarl.algorithms.atoc import ATOCConfig, ATOCGroupScheduler, ATOCLearner, ATOCPolicy
from modmarl.algorithms.atoc.algorithm import (
    ATOCActor,
    ATOCCommunicationChannel,
    ATOCReplayBatch,
)


def _small_config(**changes) -> ATOCConfig:
    values = {
        "actor_hidden_dims": (16, 8, 8, 4),
        "critic_hidden_dims": (16, 8),
        "attention_hidden_dim": 4,
        "channel_hidden_dim": 4,
        "communication_period": 2,
        "max_collaborators": 2,
        "batch_size": 2,
        "replay_capacity": 16,
    }
    values.update(changes)
    return ATOCConfig(**values)


def _batch(batch_size: int = 3, n_agents: int = 3) -> ATOCReplayBatch:
    obs = torch.randn(batch_size, n_agents, 5)
    actions = torch.rand(batch_size, n_agents, 2)
    groups = torch.zeros(batch_size, n_agents, n_agents, dtype=torch.bool)
    groups[:, 0, :2] = True
    return ATOCReplayBatch(
        obs=obs,
        actions=actions,
        rewards=torch.randn(batch_size, n_agents),
        next_obs=torch.randn_like(obs),
        dones=torch.zeros(batch_size, n_agents),
        groups=groups,
    )


def test_atoc_defaults_match_paper_settings() -> None:
    config = ATOCConfig()
    assert config.thought_dim == 128
    assert config.critic_hidden_dims == (512, 256)
    assert config.communication_period == 15
    assert config.actor_learning_rate == pytest.approx(1e-3)
    assert config.critic_learning_rate == pytest.approx(1e-3)
    assert config.attention_learning_rate == pytest.approx(1e-3)
    assert config.gamma == pytest.approx(0.96)
    assert config.tau == pytest.approx(1e-3)
    assert config.replay_capacity == 100_000
    assert config.batch_size == 2_560
    assert config.warmup_episodes == 30
    assert config.ou_theta == pytest.approx(0.15)
    assert config.ou_sigma == pytest.approx(0.2)


def test_atoc_random_normal_initialization_is_fan_in_scaled() -> None:
    torch.manual_seed(0)
    actor = ATOCActor(5, 2)
    assert float(actor.first.weight.std().detach()) == pytest.approx(5**-0.5, rel=0.1)
    assert torch.equal(actor.first.bias, torch.zeros_like(actor.first.bias))


def test_atoc_optimizers_use_their_explicit_paper_equal_defaults() -> None:
    learner = ATOCLearner(3, 5, 2, _small_config())
    assert learner.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert learner.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert learner.attention_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_signed_navigation_actions_preserve_the_original_dynamics() -> None:
    paired_env = NoisyNavigationEnv(n_agents=2, noise=0.0, seed=4)
    signed_env = SignedNoisyNavigationEnv(n_agents=2, noise=0.0, seed=4)
    paired_obs, _ = paired_env.reset(seed=9)
    signed_obs, _ = signed_env.reset(seed=9)
    assert np.array_equal(paired_obs, signed_obs)
    signed = np.array([[0.6, -0.2], [-0.4, 0.7]], dtype=np.float32)
    paired = np.array([[0.0, 0.6, 0.2, 0.0], [0.4, 0.0, 0.0, 0.7]], dtype=np.float32)
    paired_step = paired_env.step(paired)
    signed_step = signed_env.step(signed)
    assert np.allclose(paired_step[0], signed_step[0])
    assert paired_step[1:] == signed_step[1:]


def test_atoc_actor_carries_local_thought_when_no_group_is_active() -> None:
    torch.manual_seed(0)
    policy = ATOCPolicy(5, 2, _small_config())
    obs = torch.randn(1, 3, 5)
    groups = torch.zeros(1, 3, 3, dtype=torch.bool)
    actions, thoughts, integrated = policy(obs, groups)
    independent, independent_thoughts = policy.independent_actions(obs)
    assert thoughts.shape == (1, 3, 8)
    assert actions.shape == (1, 3, 2)
    assert torch.equal(integrated, thoughts)
    assert torch.allclose(actions, independent)
    assert torch.allclose(thoughts, independent_thoughts)


def test_group_scheduler_prioritizes_unselected_then_selected_then_initiators() -> None:
    scheduler = ATOCGroupScheduler(5, period=2, max_collaborators=3)
    initiators = np.array([True, False, True, False, False])
    eligible = np.ones((5, 5), dtype=bool)
    np.fill_diagonal(eligible, False)
    distances = np.array(
        [
            [0, 3, 0.1, 1, 2],
            [3, 0, 3, 3, 3],
            [0.1, 0.2, 0, 0.3, 0.4],
            [1, 3, 0.3, 0, 3],
            [2, 3, 0.4, 3, 0],
        ],
        dtype=np.float32,
    )
    groups = scheduler.step(initiators, distances, eligible)
    assert np.flatnonzero(groups[0]).tolist() == [0, 1, 3, 4]
    # Agent 1 is already selected, but still precedes the other initiator (agent 0).
    assert np.flatnonzero(groups[2]).tolist() == [1, 2, 3, 4]

    changed = scheduler.step(np.zeros(5, dtype=bool), distances, eligible)
    assert np.array_equal(changed, groups)
    refreshed = scheduler.step(np.zeros(5, dtype=bool), distances, eligible)
    assert not refreshed.any()


def test_attention_classifier_uses_the_same_hard_gate_during_collection() -> None:
    learner = ATOCLearner(3, 5, 2, _small_config())
    with torch.no_grad():
        for parameter in learner.policy.attention.parameters():
            parameter.zero_()
        learner.policy.attention.net[-1].bias.fill_(0.1)
    obs = torch.randn(3, 5)
    distances = np.ones((3, 3), dtype=np.float32)
    eligible = ~np.eye(3, dtype=bool)
    first = learner.act(
        obs,
        ATOCGroupScheduler(3, period=1, max_collaborators=2),
        distances,
        eligible,
        explore=False,
    )
    second = learner.act(
        obs,
        ATOCGroupScheduler(3, period=1, max_collaborators=2),
        distances,
        eligible,
        deterministic=True,
        explore=False,
    )
    assert torch.equal(first.groups, second.groups)
    assert first.groups.any()


def test_overlapping_groups_carry_first_integrated_thought_into_second_group() -> None:
    class GroupSum(nn.Module):
        def forward(self, sequence):
            padded, lengths = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True)
            mask = torch.arange(padded.shape[1])[None] < lengths[:, None]
            total = (padded * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            output = total.expand_as(padded)
            return (
                nn.utils.rnn.pack_padded_sequence(
                    output, lengths, batch_first=True, enforce_sorted=False,
                ),
                None,
            )

    channel = ATOCCommunicationChannel(thought_dim=2, hidden_dim=1)
    channel.channel = GroupSum()
    channel.output = nn.Identity()
    thoughts = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
    groups = torch.zeros(1, 3, 3, dtype=torch.bool)
    groups[0, 0, [0, 1]] = True
    groups[0, 2, [1, 2]] = True
    integrated = channel(thoughts, groups)
    assert torch.equal(
        integrated,
        torch.tensor([[[3.0, 0.0], [7.0, 0.0], [7.0, 0.0]]]),
    )


def test_ddpg_update_changes_actor_channel_critic_and_not_attention() -> None:
    torch.manual_seed(1)
    learner = ATOCLearner(3, 5, 2, _small_config(), action_low=0.0, action_high=1.0)
    actor_before = copy.deepcopy(learner.policy.actor.state_dict())
    channel_before = copy.deepcopy(learner.policy.channel.state_dict())
    critic_before = copy.deepcopy(learner.critic.state_dict())
    attention_before = copy.deepcopy(learner.policy.attention.state_dict())
    target_before = copy.deepcopy(learner.target_policy.state_dict())
    result = learner.update(_batch())
    assert result is not None
    assert result.critic_loss >= 0.0
    assert _state_changed(actor_before, learner.policy.actor.state_dict())
    assert _state_changed(channel_before, learner.policy.channel.state_dict())
    assert _state_changed(critic_before, learner.critic.state_dict())
    assert not _state_changed(attention_before, learner.policy.attention.state_dict())
    assert _state_changed(target_before, learner.target_policy.state_dict())


def test_soft_targets_polyak_average_batch_norm_statistics() -> None:
    learner = ATOCLearner(3, 5, 2, _small_config(tau=0.25))
    learner.target_critic.hidden_norm.running_mean.zero_()
    learner.critic.hidden_norm.running_mean.fill_(4.0)
    learner.target_critic.hidden_norm.num_batches_tracked.zero_()
    learner.critic.hidden_norm.num_batches_tracked.fill_(7)
    learner.soft_update()
    assert torch.equal(
        learner.target_critic.hidden_norm.running_mean,
        torch.ones(learner.config.critic_hidden_dims[0]),
    )
    assert learner.target_critic.hidden_norm.num_batches_tracked.item() == 7


def test_attention_episode_update_changes_only_attention_parameters() -> None:
    torch.manual_seed(2)
    learner = ATOCLearner(3, 5, 2, _small_config())
    obs = torch.randn(4, 3, 5)
    actions = torch.rand(4, 3, 2) * 2 - 1
    groups = torch.zeros(4, 3, 3, dtype=torch.bool)
    groups[:, 0, :2] = True
    groups[:, 2, 1:] = True
    attention_before = copy.deepcopy(learner.policy.attention.state_dict())
    actor_before = copy.deepcopy(learner.policy.actor.state_dict())
    channel_before = copy.deepcopy(learner.policy.channel.state_dict())
    loss = learner.update_attention_episode(obs, actions, groups)
    assert loss is not None and loss >= 0.0
    assert _state_changed(attention_before, learner.policy.attention.state_dict())
    assert not _state_changed(actor_before, learner.policy.actor.state_dict())
    assert not _state_changed(channel_before, learner.policy.channel.state_dict())


def test_attention_episode_without_a_two_agent_group_is_a_noop() -> None:
    learner = ATOCLearner(1, 5, 2, _small_config())
    loss = learner.update_attention_episode(
        torch.randn(2, 1, 5),
        torch.randn(2, 1, 2),
        torch.ones(2, 1, 1, dtype=torch.bool),
    )
    assert loss is None


def test_replay_preserves_continuous_actions_and_group_matrix() -> None:
    learner = ATOCLearner(3, 5, 2, _small_config())
    groups = np.eye(3, dtype=bool)
    learner.store_transition(
        np.zeros((3, 5), dtype=np.float32),
        np.full((3, 2), 0.25, dtype=np.float32),
        1.0,
        np.ones((3, 5), dtype=np.float32),
        False,
        groups,
    )
    batch = learner.replay.sample(1, torch.device("cpu"))
    assert torch.equal(batch.actions, torch.full((1, 3, 2), 0.25))
    assert torch.equal(batch.groups, torch.as_tensor(groups).unsqueeze(0))


def test_atoc_trainer_smoke_and_checkpoint(tmp_path) -> None:
    from examples.train_atoc import train

    checkpoint = tmp_path / "atoc.pt"
    summary = train(
        episodes=1,
        evaluation_episodes=1,
        config=_small_config(warmup_episodes=0),
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "atoc"
    assert summary["source_revision"].startswith("paper:")
    assert len(summary["returns"]) == 1
    assert 0.0 <= summary["communication_rate"] <= 1.0
    assert checkpoint.exists()


def _state_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)
