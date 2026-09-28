from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

import examples.train_mdmaddpg as train_mdmaddpg_module
from examples.train_mdmaddpg import train
from marl_envs import make_env
from modmarl import (
    MDMADDPGConfig,
    MDMADDPGLearner,
    MDMADDPGUpdate,
    MemoryDrivenActor,
    SharedMemory,
)


def test_mdmaddpg_architecture_matches_paper_dimensions() -> None:
    actor = MemoryDrivenActor(obs_dim=5, action_dim=4)
    assert actor.memory_dim == 200
    assert (actor.encoder_hidden.in_features, actor.encoder_hidden.out_features) == (5, 512)
    assert (actor.encoder_output.in_features, actor.encoder_output.out_features) == (512, 200)
    assert (actor.policy_hidden.in_features, actor.policy_hidden.out_features) == (600, 256)
    assert (actor.policy_output.in_features, actor.policy_output.out_features) == (256, 4)


def test_mdmaddpg_context_is_linear_and_read_gate_is_batched() -> None:
    memory_dim = 4
    actor = MemoryDrivenActor(obs_dim=3, action_dim=2, memory_dim=memory_dim)
    with torch.no_grad():
        actor.encoder_hidden.weight.zero_()
        actor.encoder_hidden.bias.fill_(1.0)
        actor.encoder_output.weight.zero_()
        actor.encoder_output.bias.fill_(1.0)
        actor.context.weight.copy_(-torch.eye(memory_dim))

    read_inputs: list[torch.Tensor] = []
    handle = actor.read_gate.register_forward_pre_hook(
        lambda module, args: read_inputs.append(args[0].detach().clone()),
    )
    actor(torch.zeros(2, 3), torch.tensor([[0.0] * 4, [1.0] * 4]))
    handle.remove()

    assert len(read_inputs) == 1
    assert torch.equal(read_inputs[0][:, memory_dim : 2 * memory_dim], -torch.ones(2, 4))
    assert not torch.equal(read_inputs[0][0], read_inputs[0][1])


def test_memory_driven_actor_shapes_and_writes_memory() -> None:
    actor = MemoryDrivenActor(obs_dim=5, action_dim=4, memory_dim=16)
    obs = torch.randn(6, 5)
    memory = torch.zeros(6, 16)
    logits, next_memory = actor(obs, memory)
    one_hot, action_idx, sampled_logits, written = actor.sample(obs, memory)
    assert tuple(logits.shape) == (6, 4)
    assert tuple(next_memory.shape) == (6, 16)
    assert tuple(one_hot.shape) == (6, 4)
    assert tuple(action_idx.shape) == (6,)
    assert tuple(sampled_logits.shape) == (6, 4)
    assert tuple(written.shape) == (6, 16)
    assert torch.isfinite(written).all()
    assert not torch.allclose(written, torch.zeros_like(written))


def test_mdmaddpg_config_matches_paper_training_protocol() -> None:
    config = MDMADDPGConfig()
    assert config.memory_dim == 200
    assert config.actor_learning_rate == pytest.approx(1e-4)
    assert config.critic_learning_rate == pytest.approx(1e-3)
    assert config.gamma == pytest.approx(0.95)
    assert config.tau == pytest.approx(0.01)
    assert config.replay_capacity == 1_000_000
    assert config.batch_size == 1024
    assert config.update_interval == 100


def test_mdmaddpg_validation_task_matches_papers_two_agent_horizon() -> None:
    environment = make_env("paper_mdmaddpg_navigation", 9, 7, 5)
    obs, _ = environment.reset(seed=5)
    _, reward, _, truncated, info = environment.step(np.zeros(2, dtype=np.int64))
    assert environment.n_agents == 2
    assert environment.horizon == 100
    assert obs.shape[0] == 2
    assert info["agent_rewards"].shape == (2,)
    assert reward == pytest.approx(float(info["agent_rewards"].mean()))
    assert not truncated


def test_mdmaddpg_trainer_keeps_paper_reward_vector(monkeypatch) -> None:
    captured: list[tuple[int, ...]] = []
    original = MDMADDPGLearner.store_transition

    def record(
        self, obs, actions, rewards, next_obs, dones, memory_seen, memory_written,
    ):
        captured.append(np.asarray(rewards).shape)
        return original(
            self, obs, actions, rewards, next_obs, dones,
            memory_seen, memory_written,
        )

    monkeypatch.setattr(MDMADDPGLearner, "store_transition", record)
    monkeypatch.setattr(
        train_mdmaddpg_module,
        "_evaluate",
        lambda *args, **kwargs: {
            "returns": [], "successes": [], "communication_rate": 1.0,
        },
    )
    train_mdmaddpg_module.train(
        env="paper_mdmaddpg_navigation", episodes=1, evaluation_episodes=1,
    )
    assert captured == [(2,)] * 100


def test_shared_memory_resets_to_one_fixed_random_value() -> None:
    torch.manual_seed(9)
    memory = SharedMemory(16)
    first = memory.reset(3)
    second = memory.reset(3)
    assert torch.equal(first, second)
    assert torch.equal(first[0], first[1])
    assert not torch.allclose(first, torch.zeros_like(first))


def test_mdmaddpg_rollout_threads_memory_sequentially() -> None:
    learner = MDMADDPGLearner(
        2, 5, 4, config=MDMADDPGConfig(memory_dim=8, replay_capacity=64),
    )
    learner.reset_memory()
    first_actions, first_seen, first_written = learner.act(torch.randn(2, 5))
    _second_actions, second_seen, second_written = learner.act(torch.randn(2, 5))

    assert tuple(first_actions.shape) == (2,)
    assert torch.equal(first_written[0], first_seen[1])
    assert torch.equal(first_written[-1], second_seen[0])
    assert torch.equal(second_written[0], second_seen[1])


def test_mdmaddpg_reward_normalization_is_per_agent() -> None:
    learner = MDMADDPGLearner(
        2, 3, 2,
        config=MDMADDPGConfig(memory_dim=4, replay_capacity=16, batch_size=4),
    )
    for step in range(4):
        learner.store_transition(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            np.array([step, 10 * step], dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
        )
    batch = learner._sample_normalized(torch.device("cpu"))
    assert torch.allclose(batch.rewards.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(batch.rewards.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)


def test_mdmaddpg_actor_loss_reaches_memory_gates() -> None:
    torch.manual_seed(3)
    learner = MDMADDPGLearner(
        2, 5, 4,
        config=MDMADDPGConfig(memory_dim=8, replay_capacity=16, batch_size=4),
    )
    for step in range(4):
        learner.store_transition(
            np.random.randn(2, 5).astype(np.float32),
            np.random.randint(0, 4, size=2),
            np.array([step, step + 1], dtype=np.float32),
            np.random.randn(2, 5).astype(np.float32),
            np.zeros(2, dtype=np.float32),
            np.random.randn(2, 8).astype(np.float32),
            np.random.randn(2, 8).astype(np.float32),
        )
    learner.train()
    learner._update_agent(learner._sample_normalized(torch.device("cpu")), 0)
    actor = learner.agents[0].actor
    for gate in (actor.candidate, actor.input_gate, actor.forget_gate):
        assert gate.weight.grad is not None
        assert gate.weight.grad.abs().sum().item() > 0.0


def test_mdmaddpg_updates_each_agent_every_100_steps(monkeypatch) -> None:
    learner = MDMADDPGLearner(
        2, 3, 2,
        config=MDMADDPGConfig(memory_dim=4, replay_capacity=128, batch_size=2),
    )
    marker = MDMADDPGUpdate(0, 1.0, 2.0, 3.0)
    calls: list[int] = []

    monkeypatch.setattr(learner, "_sample_normalized", lambda device: object())

    def fake_update(batch, agent_index):
        calls.append(agent_index)
        return marker

    monkeypatch.setattr(learner, "_update_agent", fake_update)
    transition = (
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
    )
    for _ in range(99):
        learner.store_transition(*transition)
        assert learner.update() == []
    learner.store_transition(*transition)
    assert learner.update() == [marker, marker]
    assert calls == [0, 1]


def test_mdmaddpg_targets_are_separate_and_gradient_free() -> None:
    learner = MDMADDPGLearner(
        2, 3, 2, config=MDMADDPGConfig(memory_dim=4),
    )
    for agent in learner.agents:
        assert agent.target_actor is not agent.actor
        assert agent.target_critic is not agent.critic
        assert all(not parameter.requires_grad for parameter in agent.target_actor.parameters())
        assert all(not parameter.requires_grad for parameter in agent.target_critic.parameters())


def test_mdmaddpg_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "mdmaddpg.pt"
    summary = train(
        env="navigation",
        n_agents=2,
        horizon=6,
        episodes=2,
        seed=5,
        memory_dim=8,
        buffer_size=64,
        batch_size=4,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "mdmaddpg"
    assert summary["env"] == "navigation"
    assert summary["config"]["memory_dim"] == 8
    assert summary["config"]["horizon"] == 6
    assert checkpoint.exists()
