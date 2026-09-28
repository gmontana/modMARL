"""Golden and integration tests for continuous DDPG."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_ddpg import _update, train
from modmarl import (
    DDPGActor,
    DDPGAgent,
    DDPGConfig,
    DDPGCritic,
    DDPGReplayBuffer,
)


def test_actor_outputs_bounded_continuous_actions() -> None:
    actor = DDPGActor(5, 4, (32, 24), action_low=0.0, action_high=1.0)
    actions = actor(torch.randn(6, 5))
    assert actions.shape == (6, 4)
    assert torch.all((0.0 <= actions) & (actions <= 1.0))


def test_critic_injects_action_after_observation_layer() -> None:
    critic = DDPGCritic(5, 4, (32, 24))
    assert critic.obs_fc.in_features == 5
    assert critic.obs_fc.out_features == 32
    assert critic.joint_fc.in_features == 32 + 4
    assert critic(torch.randn(6, 5), torch.randn(6, 4)).shape == (6,)


def test_paper_batch_normalization_contract() -> None:
    actor = DDPGActor(5, 4, (32, 24))
    critic = DDPGCritic(5, 4, (32, 24))
    assert isinstance(actor.input_norm, torch.nn.BatchNorm1d)
    assert isinstance(actor.first_norm, torch.nn.BatchNorm1d)
    assert isinstance(actor.second_norm, torch.nn.BatchNorm1d)
    assert isinstance(critic.input_norm, torch.nn.BatchNorm1d)
    assert isinstance(critic.obs_hidden_norm, torch.nn.BatchNorm1d)
    assert not hasattr(critic, "joint_norm")


def test_paper_initialization_ranges() -> None:
    actor = DDPGActor(5, 4, (32, 24))
    assert actor.fc1.weight.abs().max() <= 1 / np.sqrt(5) + 1e-7
    assert actor.fc2.weight.abs().max() <= 1 / np.sqrt(32) + 1e-7
    assert actor.output.weight.abs().max() <= 3e-3 + 1e-7
    assert actor.output.bias.abs().max() <= 3e-3 + 1e-7


def test_agent_uses_paper_optimizer_defaults() -> None:
    agent = DDPGAgent(5, 4, hidden_dims=(32, 24))
    assert agent.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert agent.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert agent.critic_optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-2)
    assert agent.gamma == pytest.approx(0.99)
    assert agent.tau == pytest.approx(0.001)


def test_agent_owns_one_canonical_configuration() -> None:
    config = DDPGConfig(
        gamma=0.7,
        tau=0.2,
        actor_learning_rate=3e-4,
        critic_learning_rate=4e-4,
        critic_weight_decay=5e-3,
        ou_theta=0.25,
        ou_sigma=0.35,
    )
    agent = DDPGAgent(5, 2, (16, 12), config=config)
    assert agent.config is config
    assert agent.gamma == pytest.approx(config.gamma)
    assert agent.tau == pytest.approx(config.tau)
    assert agent.actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        config.actor_learning_rate,
    )
    critic_group = agent.critic_optimizer.param_groups[0]
    assert critic_group["lr"] == pytest.approx(config.critic_learning_rate)
    assert critic_group["weight_decay"] == pytest.approx(config.critic_weight_decay)
    assert agent._noise.theta == pytest.approx(config.ou_theta)
    assert agent._noise.sigma == pytest.approx(config.ou_sigma)


def test_config_matches_paper_schedule() -> None:
    config = DDPGConfig()
    assert config.batch_size == 64
    assert config.replay_capacity == 1_000_000
    assert not config.update_due(63)
    assert config.update_due(64)


def test_ou_exploration_is_seeded_and_episode_resettable() -> None:
    torch.manual_seed(0)
    first = DDPGAgent(5, 2, (16, 12), seed=9)
    torch.manual_seed(0)
    second = DDPGAgent(5, 2, (16, 12), seed=9)
    obs = torch.zeros(1, 5)
    first_action = first.act(obs)
    second_action = second.act(obs)
    torch.testing.assert_close(first_action, second_action)
    first.reset_noise()
    second.reset_noise()
    torch.testing.assert_close(first.act(obs), second.act(obs))


def test_deterministic_action_has_no_exploration_noise() -> None:
    agent = DDPGAgent(5, 2, (16, 12), action_low=0.0, action_high=1.0)
    obs = torch.randn(3, 5)
    agent.actor.eval()
    torch.testing.assert_close(agent.act(obs, explore=False), agent.actor(obs))


def test_ou_sigma_is_measured_in_environment_action_units(monkeypatch) -> None:
    agent = DDPGAgent(5, 2, (16, 12), action_low=-10.0, action_high=10.0)
    obs = torch.randn(1, 5)
    deterministic = agent.act(obs, explore=False)
    monkeypatch.setattr(
        agent._noise,
        "sample",
        lambda: np.asarray([0.2, -0.3], dtype=np.float32),
    )
    torch.testing.assert_close(
        agent.act(obs, explore=True),
        deterministic + torch.tensor([[0.2, -0.3]]),
    )


def test_act_uses_running_batchnorm_statistics_for_batch_one() -> None:
    agent = DDPGAgent(5, 2, (16, 12))
    before = agent.actor.input_norm.running_mean.clone()
    action = agent.act(torch.randn(1, 5), explore=False)
    assert action.shape == (1, 2)
    torch.testing.assert_close(agent.actor.input_norm.running_mean, before)
    assert agent.actor.training


def test_replay_preserves_continuous_action_vectors() -> None:
    replay = DDPGReplayBuffer(8, n_agents=2, obs_dim=3, action_dim=4)
    actions = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    replay.add(np.zeros((2, 3)), actions, 1.0, np.ones((2, 3)), False)
    batch = replay.sample(1, torch.device("cpu"))
    assert batch.actions.shape == (1, 2, 4)
    torch.testing.assert_close(batch.actions[0], torch.as_tensor(actions))


def test_update_rejects_quantized_actions() -> None:
    agent = DDPGAgent(5, 4, (16, 12))
    with pytest.raises(ValueError, match="continuous vectors"):
        agent.update(
            torch.randn(8, 5),
            torch.randint(4, (8,)),
            torch.randn(8),
            torch.randn(8, 5),
            torch.zeros(8),
        )


def test_td_target_uses_deterministic_target_actor(monkeypatch) -> None:
    class ConstantCritic(torch.nn.Module):
        def forward(self, obs, actions):
            return torch.full((obs.shape[0],), 3.0)

    agent = DDPGAgent(
        5, 4, (16, 12), config=DDPGConfig(critic_weight_decay=0.0),
    )
    agent.target_critic = ConstantCritic()
    monkeypatch.setattr(agent, "soft_update", lambda *args, **kwargs: None)
    captured = []
    real_mse = torch.nn.functional.mse_loss

    def spy(prediction, target):
        captured.append(target.detach().clone())
        return real_mse(prediction, target)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", spy)
    rewards = torch.tensor([1.0, -2.0, 0.5, 3.0])
    dones = torch.tensor([1.0, 0.0, 1.0, 0.0])
    agent.update(
        torch.randn(4, 5),
        torch.randn(4, 4),
        rewards,
        torch.randn(4, 5),
        dones,
    )
    expected = rewards + 0.99 * (1.0 - dones) * 3.0
    torch.testing.assert_close(captured[0], expected)


def test_actor_update_does_not_accumulate_critic_gradients() -> None:
    agent = DDPGAgent(5, 4, (16, 12))
    agent.update(
        torch.randn(8, 5),
        torch.randn(8, 4),
        torch.randn(8),
        torch.randn(8, 5),
        torch.zeros(8),
    )
    assert all(parameter.grad is None for parameter in agent.critic.parameters())
    assert all(parameter.requires_grad for parameter in agent.critic.parameters())


def test_soft_update_interpolates_targets() -> None:
    agent = DDPGAgent(5, 4, (16, 12))
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.fill_(4.0)
        for parameter in agent.target_actor.parameters():
            parameter.zero_()
    agent.soft_update(0.25)
    for parameter in agent.target_actor.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 1.0))


def test_soft_update_interpolates_batchnorm_running_statistics() -> None:
    agent = DDPGAgent(5, 4, (16, 12))
    with torch.no_grad():
        agent.actor.input_norm.running_mean.fill_(4.0)
        agent.target_actor.input_norm.running_mean.zero_()
    agent.soft_update(0.25)
    torch.testing.assert_close(
        agent.target_actor.input_norm.running_mean,
        torch.ones_like(agent.target_actor.input_norm.running_mean),
    )


def test_update_routes_each_agents_continuous_slice() -> None:
    agents = [DDPGAgent(5, 4, (16, 12)) for _ in range(2)]
    replay = DDPGReplayBuffer(8, 2, 5, 4)
    for _ in range(4):
        replay.add(
            np.random.randn(2, 5),
            np.random.uniform(-1.0, 1.0, (2, 4)),
            1.0,
            np.random.randn(2, 5),
            False,
        )
    updates = _update(agents, replay.sample(4, torch.device("cpu")))
    assert len(updates) == 2
    assert all(np.isfinite(update.critic_loss) for update in updates)


def test_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "ddpg.pt"
    summary = train(
        n_agents=2,
        horizon=4,
        episodes=2,
        seed=5,
        hidden_dims=(16, 12),
        buffer_size=32,
        batch_size=4,
        evaluation_episodes=1,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "ddpg"
    assert summary["env"] == "noisy_navigation"
    assert checkpoint.exists()
