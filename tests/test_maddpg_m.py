from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("gymnasium")

import examples.train_maddpg_m as train_maddpg_m
from examples.train_maddpg_m import _target_medium, _update, train
from marl_envs.noisy_navigation import NoisyNavigationEnv
from modmarl import MADDPGMAgent
from modmarl.common.replay import MADDPGMReplayBatch, MADDPGMReplayBuffer


def test_noisy_navigation_env_shapes() -> None:
    env = NoisyNavigationEnv(n_agents=2, horizon=5, gifted_agent=0, seed=3)
    obs, info = env.reset(seed=3)
    assert obs.shape == (2, env.obs_dim)
    assert "mean_distance" in info
    next_obs, reward, _terminated, _truncated, _ = env.step(torch.rand(2, 4).numpy())
    assert next_obs.shape == (2, env.obs_dim)
    assert isinstance(reward, float)
    assert isinstance(env.intrinsic_reward(0), float)
    env.close()


def test_noisy_navigation_uses_paper_observation_and_continuous_action() -> None:
    env = NoisyNavigationEnv(n_agents=3, seed=8)
    obs, _ = env.reset(seed=8)
    assert env.num_actions == 4
    assert env.obs_dim == 4 + 2 * 3 + 2 * 2
    assert torch.equal(torch.from_numpy(obs[:, :2]), torch.zeros(3, 2))
    next_obs, *_ = env.step(torch.rand(3, 4).numpy())
    assert next_obs.shape == obs.shape


def test_maddpg_m_replay_preserves_continuous_actions() -> None:
    replay = MADDPGMReplayBuffer(capacity=2, n_agents=3, obs_dim=4, action_dim=4)
    action = torch.rand(3, 4).numpy()
    replay.add(
        obs=torch.zeros(3, 4).numpy(), comm_actions=torch.zeros(3).numpy(),
        medium=torch.zeros(4).numpy(), actions=action, ext_reward=0.0, int_reward=0.0,
        next_obs=torch.zeros(3, 4).numpy(), done=False,
    )
    assert replay.actions.shape == (2, 3, 4)
    assert torch.equal(torch.from_numpy(replay.actions[0]), torch.from_numpy(action))


def test_maddpg_m_agent_shapes() -> None:
    agent = MADDPGMAgent(n_agents=2, obs_dim=6, action_dim=5, hidden_dim=32, critic_hidden_dim=48)
    obs = torch.randn(4, 6)
    medium = torch.randn(4, 6)
    willingness = agent.comm_policy(obs)
    assert tuple(willingness.shape) == (4, 1)
    assert torch.all((willingness >= 0.0) & (willingness <= 1.0))
    action = agent.action_policy(obs, medium)
    assert tuple(action.shape) == (4, 5)
    assert torch.all((action >= 0.0) & (action <= 1.0))
    agent.soft_update(0.01)


def test_maddpg_m_released_network_widths() -> None:
    agent = MADDPGMAgent(n_agents=3, obs_dim=8, action_dim=5)
    comm_linear = [layer for layer in agent.comm_policy.modules() if isinstance(layer, torch.nn.Linear)]
    critic_linear = [layer for layer in agent.comm_critic.modules() if isinstance(layer, torch.nn.Linear)]
    assert [layer.out_features for layer in comm_linear] == [64, 64, 1]
    assert [layer.out_features for layer in critic_linear] == [128, 128, 1]


def test_maddpg_m_target_medium_comes_from_next_obs() -> None:
    class FirstFeaturePolicy(torch.nn.Module):
        def forward(self, obs):
            return obs[:, :1]

    agents = [
        SimpleNamespace(target_comm_policy=FirstFeaturePolicy()),
        SimpleNamespace(target_comm_policy=FirstFeaturePolicy()),
    ]
    next_obs = torch.tensor(
        [
            [[0.1, 1.0, 1.1], [0.9, 2.0, 2.2]],
            [[0.8, 3.0, 3.3], [0.2, 4.0, 4.4]],
        ]
    )

    medium = _target_medium(agents, next_obs)

    expected = torch.stack([next_obs[0, 1], next_obs[1, 0]])
    assert torch.equal(medium, expected)


def test_maddpg_m_update_routes_rewards_to_levels(monkeypatch) -> None:
    # With every critic stubbed to zero (and dones=0), the action-level regression target
    # must be exactly int_rewards and the comm-level target exactly ext_rewards: the two
    # policy levels train on different reward streams.
    class ZeroCritic(torch.nn.Module):
        """Returns zeros but stays grad-connected so every backward in _update is valid."""

        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, first, *rest):
            return torch.zeros(first.shape[0]) + 0.0 * self.dummy

    torch.manual_seed(0)
    n_agents, obs_dim, num_actions, batch_size = 2, 4, 3, 5
    agents = [
        MADDPGMAgent(n_agents, obs_dim, num_actions, hidden_dim=32, critic_hidden_dim=32)
        for _ in range(n_agents)
    ]
    for agent in agents:
        agent.action_critic = ZeroCritic()
        agent.target_action_critic = ZeroCritic()
        agent.comm_critic = ZeroCritic()
        agent.target_comm_critic = ZeroCritic()

    captured: list[torch.Tensor] = []
    real_mse_loss = torch.nn.functional.mse_loss

    def spy_mse_loss(pred, target, *args, **kwargs):
        captured.append(target.detach().clone())
        return real_mse_loss(pred, target, *args, **kwargs)

    monkeypatch.setattr(train_maddpg_m.F, "mse_loss", spy_mse_loss)

    batch = MADDPGMReplayBatch(
        obs=torch.randn(batch_size, n_agents, obs_dim),
        comm_actions=torch.rand(batch_size, n_agents),
        medium=torch.randn(batch_size, obs_dim),
        actions=torch.rand(batch_size, n_agents, num_actions),
        ext_rewards=torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
        int_rewards=torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0]),
        next_obs=torch.randn(batch_size, n_agents, obs_dim),
        dones=torch.zeros(batch_size),
    )
    comm_policy_opts = [torch.optim.Adam(a.comm_policy.parameters(), lr=1e-3) for a in agents]
    action_policy_opts = [torch.optim.Adam(a.action_policy.parameters(), lr=1e-3) for a in agents]
    comm_critic_opts = [torch.optim.Adam(a.comm_critic.parameters(), lr=1e-3) for a in agents]
    action_critic_opts = [torch.optim.Adam(a.action_critic.parameters(), lr=1e-3) for a in agents]
    _update(
        agents, comm_policy_opts, action_policy_opts, comm_critic_opts, action_critic_opts,
        batch, batch, num_actions,
    )

    # Per agent, _update regresses the action critic first, then the comm critic.
    assert len(captured) == 2 * n_agents
    for agent_id in range(n_agents):
        assert torch.equal(captured[2 * agent_id], batch.int_rewards)
        assert torch.equal(captured[2 * agent_id + 1], batch.ext_rewards)


def test_maddpg_m_action_target_keeps_replayed_medium() -> None:
    class RecordingPolicy(torch.nn.Module):
        def __init__(self, action_dim: int) -> None:
            super().__init__()
            self.action_dim = action_dim
            self.mediums: list[torch.Tensor] = []

        def forward(self, obs, medium):
            self.mediums.append(medium.detach().clone())
            return torch.zeros(obs.shape[0], self.action_dim)

    torch.manual_seed(4)
    n_agents, obs_dim, action_dim, batch_size = 2, 4, 3, 5
    agents = [MADDPGMAgent(n_agents, obs_dim, action_dim, 16, 16) for _ in range(n_agents)]
    recorders = [RecordingPolicy(action_dim) for _ in range(n_agents)]
    for agent, recorder in zip(agents, recorders):
        agent.target_action_policy = recorder
    def opts(modules):
        return [torch.optim.Adam(module.parameters(), lr=1e-3) for module in modules]
    batch = MADDPGMReplayBatch(
        obs=torch.randn(batch_size, n_agents, obs_dim),
        comm_actions=torch.rand(batch_size, n_agents),
        medium=torch.full((batch_size, obs_dim), 7.0),
        actions=torch.rand(batch_size, n_agents, action_dim),
        ext_rewards=torch.randn(batch_size),
        int_rewards=torch.randn(batch_size),
        next_obs=torch.randn(batch_size, n_agents, obs_dim),
        dones=torch.zeros(batch_size),
    )
    _update(
        agents,
        opts([agent.comm_policy for agent in agents]),
        opts([agent.action_policy for agent in agents]),
        opts([agent.comm_critic for agent in agents]),
        opts([agent.action_critic for agent in agents]),
        batch,
        batch,
        action_dim,
    )
    assert all(torch.equal(recorder.mediums[0], batch.medium) for recorder in recorders)


def test_maddpg_m_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "maddpg_m.pt"
    summary = train(
        n_agents=2,
        horizon=6,
        episodes=2,
        seed=5,
        hidden_dim=32,
        critic_hidden_dim=32,
        communication_interval=2,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "maddpg_m"
    assert summary["env"] == "noisy_navigation"
    assert len(summary["initial_evaluation"]["returns"]) == 2
    assert len(summary["random_evaluation"]["returns"]) == 2
    assert len(summary["final_evaluation"]["returns"]) == 2
    assert summary["communication_rate"] == 0.5
    assert summary["config"]["communication_interval"] == 2
    assert checkpoint.exists()


def test_maddpg_m_script_runs_directly(tmp_path) -> None:
    checkpoint = tmp_path / "direct.pt"
    subprocess.run(
        [
            sys.executable,
            "examples/train_maddpg_m.py",
            "--episodes", "1",
            "--n-agents", "3",
            "--checkpoint", str(checkpoint),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint.exists()


def test_vendored_particle_scenarios_compile() -> None:
    scenario_dir = Path(__file__).resolve().parent.parent / "marl_envs/vendors/multiagentsha/scenarios"
    for source in scenario_dir.glob("*.py"):
        py_compile.compile(source, doraise=True)
