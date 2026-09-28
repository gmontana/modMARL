from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("gymnasium")

from examples.train_intention_sharing import train
from modmarl.algorithms.intention_sharing import (
    IntentionSharingConfig,
    IntentionSharingLearner,
    IntentionSharingPolicy,
)


def _policy(agent_index: int = 0) -> IntentionSharingPolicy:
    return IntentionSharingPolicy(
        agent_index=agent_index,
        n_agents=3,
        obs_dim=5,
        action_dim=4,
        message_dim=3,
        hidden_dim=32,
    )


def test_intention_sharing_policy_is_one_agent_wide() -> None:
    policy = _policy(agent_index=1)
    obs = torch.randn(6, 5)
    incoming = torch.zeros(6, 3, 3)
    logits, outgoing = policy(obs, incoming)
    one_hot, action_index, sampled_logits, sampled_outgoing = policy.sample(obs, incoming)
    assert logits.shape == sampled_logits.shape == (6, 4)
    assert outgoing.shape == sampled_outgoing.shape == (6, 3)
    assert one_hot.shape == (6, 4)
    assert action_index.shape == (6,)


def test_received_messages_condition_policy_and_temporal_attention() -> None:
    torch.manual_seed(0)
    policy = _policy()
    obs = torch.randn(2, 5)
    incoming = torch.randn(2, 3, 3)
    action = F.one_hot(torch.tensor([1, 2]), 4).float()
    base_logits, _ = policy(obs, incoming)
    base_message = policy.messages(obs, incoming, action)
    changed = incoming.clone()
    changed[:, 1] += 2.0
    changed_logits, _ = policy(obs, changed)
    changed_message = policy.messages(obs, changed, action)
    assert not torch.allclose(changed_logits, base_logits)
    assert not torch.allclose(changed_message, base_message)


def test_first_imagined_pair_contains_the_executed_action() -> None:
    policy = _policy()
    obs = torch.randn(2, 5)
    incoming = torch.zeros(2, 3, 3)
    executed = F.one_hot(torch.tensor([3, 1]), 4).float()
    trajectory, _, _ = policy.imagine(obs, incoming, executed)
    torch.testing.assert_close(trajectory[:, 0, :5], obs)
    torch.testing.assert_close(trajectory[:, 0, 5:], executed)


def test_imagination_is_a_raw_observation_residual() -> None:
    torch.manual_seed(0)
    policy = _policy()
    obs = torch.randn(2, 5)
    incoming = torch.zeros(2, 3, 3)
    executed = F.one_hot(torch.tensor([0, 1]), 4).float()
    final = [
        module
        for module in policy.dynamics_model.modules()
        if isinstance(module, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()
    trajectory, _, predicted = policy.imagine(obs, incoming, executed)
    torch.testing.assert_close(predicted, obs)
    for step in range(policy.imagination_horizon):
        torch.testing.assert_close(trajectory[:, step, :5], obs)


def test_action_predictor_excludes_self_and_uses_equation_twelve_mse() -> None:
    torch.manual_seed(1)
    policy = _policy(agent_index=1)
    obs = torch.randn(2, 5)
    incoming = torch.zeros(2, 3, 3)
    actions = torch.tensor([[0, 1, 2], [3, 2, 1]])
    own = F.one_hot(actions[:, 1], 4).float()
    _, other_logits, predicted_next = policy.imagine(obs, incoming, own)
    target_other = F.one_hot(torch.tensor([[0, 2], [3, 1]]), 4).float()
    expected_action_loss = F.mse_loss(torch.softmax(other_logits, -1), target_other)
    loss = policy.model_loss(obs, predicted_next.detach(), actions, incoming)
    torch.testing.assert_close(loss, expected_action_loss)


def test_equation_eleven_updates_attention_but_not_rollout_models() -> None:
    torch.manual_seed(2)
    config = IntentionSharingConfig(batch_size=2, minimum_replay_size=2)
    learner = IntentionSharingLearner(
        3, 5, 4, message_dim=3, hidden_dim=32, config=config,
    )
    obs = torch.randn(2, 3, 5)
    next_obs = torch.randn_like(obs)
    incoming = torch.randn(2, 3, 3)
    actions = F.one_hot(torch.randint(0, 4, (2, 3)), 4).float()
    written = learner.messages_for_actions(
        obs, incoming, actions, detach_trajectory=True,
    )
    receiver = learner.agents[1]
    receiver_action = learner._action_with_detached_parameters(
        receiver.policy, next_obs[:, 1], written,
    )
    baseline = torch.zeros_like(actions)
    joint = torch.stack(
        [receiver_action if index == 1 else baseline[:, index] for index in range(3)],
        dim=1,
    )
    (-receiver.critic(next_obs, joint).mean()).backward()

    sender = learner.agents[0].policy
    attention_modules = [
        sender.message_query, sender.trajectory_key, sender.trajectory_value,
    ]
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for module in attention_modules
        for parameter in module.parameters()
    )
    assert all(parameter.grad is None for parameter in sender.action_model.parameters())
    assert all(parameter.grad is None for parameter in sender.dynamics_model.parameters())
    assert all(parameter.grad is None for parameter in receiver.policy.policy_head.parameters())


def test_every_agent_owns_independent_actor_critic_and_frozen_targets() -> None:
    learner = IntentionSharingLearner(3, 5, 4, message_dim=3, hidden_dim=32)
    actor_pointers = {
        next(agent.policy.parameters()).data_ptr() for agent in learner.agents
    }
    critic_pointers = {
        next(agent.critic.parameters()).data_ptr() for agent in learner.agents
    }
    assert len(actor_pointers) == len(critic_pointers) == 3
    for agent in learner.agents:
        assert not any(parameter.requires_grad for parameter in agent.target_policy.parameters())
        assert not any(parameter.requires_grad for parameter in agent.target_critic.parameters())


def test_replay_preserves_individual_rewards_and_dones() -> None:
    config = IntentionSharingConfig(
        batch_size=1, replay_capacity=4, minimum_replay_size=1,
    )
    learner = IntentionSharingLearner(3, 5, 4, message_dim=3, hidden_dim=32, config=config)
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dones = np.array([False, True, False])
    learner.store(
        obs=np.zeros((3, 5), dtype=np.float32),
        actions=np.array([0, 1, 2]),
        rewards=rewards,
        next_obs=np.ones((3, 5), dtype=np.float32),
        dones=dones,
        messages_seen=np.zeros((3, 3), dtype=np.float32),
        messages_written=np.ones((3, 3), dtype=np.float32),
    )
    batch = learner.replay.sample(1, torch.device("cpu"))
    torch.testing.assert_close(batch.rewards[0], torch.from_numpy(rewards))
    torch.testing.assert_close(batch.dones[0], torch.from_numpy(dones.astype(np.float32)))


def test_joint_update_changes_every_independent_actor_and_critic() -> None:
    torch.manual_seed(4)
    np.random.seed(4)
    config = IntentionSharingConfig(
        batch_size=4, replay_capacity=16, minimum_replay_size=4,
    )
    learner = IntentionSharingLearner(3, 5, 4, message_dim=3, hidden_dim=32, config=config)
    for _ in range(4):
        learner.store(
            obs=np.random.randn(3, 5).astype(np.float32),
            actions=np.random.randint(0, 4, size=3),
            rewards=np.random.randn(3).astype(np.float32),
            next_obs=np.random.randn(3, 5).astype(np.float32),
            dones=np.zeros(3, dtype=np.float32),
            messages_seen=np.random.randn(3, 3).astype(np.float32),
            messages_written=np.random.randn(3, 3).astype(np.float32),
        )
    actors_before = [next(agent.policy.parameters()).detach().clone() for agent in learner.agents]
    critics_before = [next(agent.critic.parameters()).detach().clone() for agent in learner.agents]
    update = learner.update()
    assert np.isfinite(update.policy_loss)
    assert np.isfinite(update.critic_loss)
    assert all(
        not torch.equal(before, next(agent.policy.parameters()))
        for before, agent in zip(actors_before, learner.agents, strict=True)
    )
    assert all(
        not torch.equal(before, next(agent.critic.parameters()))
        for before, agent in zip(critics_before, learner.agents, strict=True)
    )


def test_intention_sharing_train_smoke_records_complete_evaluation(tmp_path) -> None:
    checkpoint = tmp_path / "intention_sharing.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        message_dim=3,
        hidden_dim=32,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        update_interval=1,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "intention_sharing"
    assert summary["config"]["parameter_sharing"] is False
    assert summary["config"]["update_interval"] == 1
    assert summary["communication_rate"] == 1.0
    assert len(summary["initial_evaluation"]["returns"]) == 2
    assert len(summary["random_evaluation"]["returns"]) == 2
    assert len(summary["final_evaluation"]["returns"]) == 2
    assert checkpoint.exists()


def test_paper_table_two_defaults() -> None:
    config = IntentionSharingConfig()
    assert config.gamma == 0.99
    assert config.learning_rate == 5e-4
    assert config.batch_size == 128
    assert config.replay_capacity == 200_000
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(train).parameters.items()
    }
    assert defaults["update_interval"] == 100
    assert _policy().imagination_horizon == 5
    assert _policy().message_dim == 3
