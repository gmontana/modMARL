from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_maddpg import _select_actions, train
from marl_envs import PaperParticleEnv, paper_particle_env_available
from modmarl import MADDPGAgent, MADDPGConfig, MADDPGLearner, MADDPGReplayBatch, MADDPGReplayBuffer
from tools.reference.compare_openai_maddpg import compare_curves


def _batch(batch_size: int = 8, n_agents: int = 3, obs_dim: int = 5, action_dim: int = 4):
    actions = torch.softmax(torch.randn(batch_size, n_agents, action_dim), dim=-1)
    return MADDPGReplayBatch(
        obs=torch.randn(batch_size, n_agents, obs_dim),
        actions=actions,
        rewards=torch.randn(batch_size, n_agents),
        next_obs=torch.randn(batch_size, n_agents, obs_dim),
        dones=torch.zeros(batch_size, n_agents),
    )


def test_maddpg_actor_deterministic_sample_matches_argmax() -> None:
    agent = MADDPGAgent(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=32)
    obs = torch.randn(6, 5)
    logits = agent.actor(obs)
    action, action_idx = agent.act(obs, deterministic=True)
    expected_idx = logits.argmax(dim=-1)
    assert torch.equal(action_idx, expected_idx)
    assert torch.equal(action, torch.nn.functional.one_hot(expected_idx, 4).float())


def test_maddpg_rollout_preserves_soft_gumbel_vectors_for_replay() -> None:
    torch.manual_seed(3)
    learner = MADDPGLearner(2, 5, 4, 32)
    vectors, indices = _select_actions(learner, np.zeros((2, 5), dtype=np.float32), torch.device("cpu"))
    assert vectors.shape == (2, 4)
    assert np.allclose(vectors.sum(axis=-1), 1.0)
    assert np.any((vectors > 0.0) & (vectors < 1.0))
    assert np.array_equal(indices, vectors.argmax(axis=-1))


def test_maddpg_replay_keeps_individual_rewards_and_action_vectors() -> None:
    replay = MADDPGReplayBuffer(8, 2, 3, 4)
    actions = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    replay.add(np.zeros((2, 3)), actions, np.asarray([1.0, -2.0]), np.ones((2, 3)), [False, True])
    batch = replay.sample(1, torch.device("cpu"))
    assert torch.allclose(batch.actions[0], torch.from_numpy(actions))
    assert torch.equal(batch.rewards[0], torch.tensor([1.0, -2.0]))
    assert torch.equal(batch.dones[0], torch.tensor([0.0, 1.0]))


def test_paper_mpe_accepts_reference_soft_action_vectors() -> None:
    if not paper_particle_env_available():
        pytest.skip("archived paper MPE dependencies are unavailable")
    env = PaperParticleEnv("paper_navigation", seed=3)
    env.reset(seed=3)
    actions = np.full((env.n_agents, env.num_actions), 1.0 / env.num_actions, dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(actions)
    assert obs.shape == (env.n_agents, env.obs_dim)
    assert np.isfinite(reward)
    assert not terminated and not truncated
    env.close()


def test_maddpg_paper_env_restores_openai_collaborative_reward() -> None:
    if not paper_particle_env_available():
        pytest.skip("archived paper MPE dependencies are unavailable")
    individual = PaperParticleEnv("paper_navigation", seed=9)
    collaborative = PaperParticleEnv("paper_maddpg_navigation", seed=9)
    individual.reset(seed=9)
    collaborative.reset(seed=9)
    actions = np.eye(individual.num_actions, dtype=np.float32)[[0] * individual.n_agents]
    _, mean_reward, _, _, _ = individual.step(actions)
    _, team_reward, _, _, _ = collaborative.step(actions)
    assert team_reward == pytest.approx(individual.n_agents * mean_reward)
    individual.close()
    collaborative.close()


def test_maddpg_critic_is_sensitive_to_each_agents_action() -> None:
    torch.manual_seed(0)
    agent = MADDPGAgent(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=32)
    batch = _batch()
    baseline = agent.critic(batch.obs, batch.actions)
    for j in range(3):
        perturbed = batch.actions.clone()
        perturbed[:, j] = torch.roll(perturbed[:, j], shifts=1, dims=-1)
        assert not torch.allclose(agent.critic(batch.obs, perturbed), baseline)


def test_maddpg_update_changes_actor_critic_and_targets() -> None:
    torch.manual_seed(0)
    agents = [MADDPGAgent(3, 5, 4, 32) for _ in range(3)]
    learner = agents[1]
    actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in learner.critic.parameters()]
    target_before = [parameter.detach().clone() for parameter in learner.target_actor.parameters()]
    diagnostics = learner.update(agents, 1, _batch())
    assert diagnostics.critic_loss >= 0.0
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, learner.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, learner.critic.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(target_before, learner.target_actor.parameters()))


def test_maddpg_td_target_uses_every_target_actor() -> None:
    class FixedTargetActor(torch.nn.Module):
        def __init__(self, action: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("action", action)

        def sample(self, obs, temperature=1.0, hard=False, deterministic=False):
            del temperature, hard, deterministic
            action = self.action.expand(obs.shape[0], -1)
            return action, action.argmax(dim=-1), action

    class CapturingTargetCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_actions = None

        def forward(self, obs, actions):
            self.seen_actions = actions.detach().clone()
            return torch.zeros(obs.shape[0])

    agents = [MADDPGAgent(2, 5, 4, 32) for _ in range(2)]
    expected = [torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0.4, 0.3, 0.2, 0.1])]
    for agent, action in zip(agents, expected):
        agent.target_actor = FixedTargetActor(action)
    capture = CapturingTargetCritic()
    agents[0].target_critic = capture
    batch = _batch(n_agents=2)
    agents[0].update(agents, 0, batch)
    expected_joint = torch.stack(expected).expand(batch.obs.shape[0], -1, -1)
    assert torch.equal(capture.seen_actions, expected_joint)


def test_maddpg_update_uses_the_learning_agents_reward() -> None:
    torch.manual_seed(4)
    config = MADDPGConfig(gamma=0.0)
    agents = [MADDPGAgent(2, 5, 4, 32, config) for _ in range(2)]
    batch = _batch(n_agents=2)
    batch.rewards[:, 0] = -7.0
    batch.rewards[:, 1] = 3.0
    result = agents[1].update(agents, 1, batch)
    assert result.reward_mean == pytest.approx(3.0)
    assert result.target_q_mean == pytest.approx(3.0)


def test_maddpg_reference_hyperparameters_are_explicit() -> None:
    config = MADDPGConfig()
    assert config.learning_rate == 1e-2
    assert config.gamma == 0.95
    assert config.tau == 0.01
    assert config.policy_regularization == 1e-3
    assert config.gradient_clip == 0.5
    assert config.batch_size == 1024
    assert config.update_interval == 100


def test_maddpg_targets_start_as_exact_online_copies() -> None:
    agent = MADDPGAgent(3, 5, 4, 32)
    for online, target in zip(agent.actor.parameters(), agent.target_actor.parameters()):
        assert torch.equal(online, target)
        assert not target.requires_grad
    for online, target in zip(agent.critic.parameters(), agent.target_critic.parameters()):
        assert torch.equal(online, target)
        assert not target.requires_grad


def test_maddpg_reference_update_schedule() -> None:
    config = MADDPGConfig(batch_size=2, max_episode_len=3, replay_capacity=8, update_interval=4)
    learner = MADDPGLearner(2, 3, 4, 16, config)
    obs = np.zeros((2, 3), dtype=np.float32)
    actions = np.full((2, 4), 0.25, dtype=np.float32)
    for _ in range(5):
        learner.store_transition(obs, actions, 0.0, obs, False)
    assert not learner.ready_to_update()
    learner.store_transition(obs, actions, 0.0, obs, False)
    assert not learner.ready_to_update()
    for _ in range(2):
        learner.store_transition(obs, actions, 0.0, obs, False)
    assert learner.ready_to_update()


def test_maddpg_reference_comparison_uses_equal_episode_windows() -> None:
    result = compare_curves(np.asarray([1.0, 3.0, 5.0, 7.0]), np.asarray([2.0, 6.0]))
    assert result["window_episodes"] == 2
    assert result["modmarl_team_returns"] == [2.0, 6.0]
    assert result["pearson_correlation"] == pytest.approx(1.0)


def test_maddpg_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "maddpg.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        hidden_dim=32,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        update_interval=1,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "maddpg"
    assert summary["updates"] > 0
    assert checkpoint.exists()
    payload = torch.load(checkpoint, weights_only=False)
    assert len(payload["actor_optimizers"]) == 3
    assert len(payload["critic_optimizers"]) == 3
