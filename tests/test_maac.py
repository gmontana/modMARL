from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("gymnasium")

import examples.train_maac as train_maac_module
from examples.train_maac import train
from marl_envs import make_env
from modmarl import MAACAgent, MAACConfig, MAACLearner, MAACUpdate
from modmarl.algorithms import maac as maac_module
from modmarl.algorithms.maac import AttentionCritic, MAACCriticOutput
from modmarl.common.replay import ReplayBatch


def test_maac_actor_deterministic_sample_matches_argmax() -> None:
    agent = MAACAgent(obs_dim=5, action_dim=4, hidden_dim=32)
    agent.eval()
    obs = torch.randn(6, 5)
    logits = agent.actor(obs)
    one_hot, action_idx, sampled_logits, _, log_probs, chosen_log_prob, _ = (
        agent.actor.sample(obs, deterministic=True)
    )
    expected_idx = logits.argmax(dim=-1)
    expected_one_hot = F.one_hot(expected_idx, num_classes=4).to(dtype=logits.dtype)
    expected_log_prob = log_probs.gather(-1, expected_idx.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(action_idx, expected_idx)
    assert torch.equal(one_hot, expected_one_hot)
    assert torch.allclose(sampled_logits, logits)
    assert torch.allclose(chosen_log_prob, expected_log_prob)


def test_maac_counterfactual_baseline_enumerates_own_actions() -> None:
    torch.manual_seed(0)
    n, obs_dim, action_dim, batch = 3, 4, 3, 5
    critic = AttentionCritic(
        n_agents=n, obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=8, attend_heads=2,
    )
    critic.eval()
    obs = torch.randn(batch, n, obs_dim)
    joint = F.one_hot(torch.randint(0, action_dim, (batch, n)), action_dim).float()
    probs = torch.softmax(torch.randn(batch, n, action_dim), dim=-1)

    with torch.no_grad():
        output = critic(obs, joint)
        for agent_id in range(n):
            baseline = (output.all_q[:, agent_id] * probs[:, agent_id]).sum(dim=-1)
            expected = torch.zeros(batch)
            for action_id in range(action_dim):
                substituted = joint.clone()
                substituted[:, agent_id] = F.one_hot(
                    torch.full((batch,), action_id), action_dim,
                ).float()
                q_substituted = critic(obs, substituted).q_taken[:, agent_id]
                assert torch.allclose(
                    output.all_q[:, agent_id, action_id], q_substituted, atol=1e-6,
                )
                expected += probs[:, agent_id, action_id] * q_substituted
            assert torch.allclose(baseline, expected, atol=1e-6)


def test_maac_attention_excludes_own_action_and_regularizes_logits() -> None:
    torch.manual_seed(1)
    n, obs_dim, action_dim, batch = 3, 4, 3, 4
    critic = AttentionCritic(
        n_agents=n, obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=8, attend_heads=2,
    )
    critic.eval()
    obs = torch.randn(batch, n, obs_dim)
    joint = F.one_hot(torch.zeros(batch, n, dtype=torch.long), action_dim).float()

    with torch.no_grad():
        output = critic(obs, joint, return_attention=True)
        own_changed = joint.clone()
        own_changed[:, 0] = F.one_hot(torch.ones(batch, dtype=torch.long), action_dim).float()
        own_output = critic(obs, own_changed)
        other_changed = joint.clone()
        other_changed[:, 1] = F.one_hot(torch.ones(batch, dtype=torch.long), action_dim).float()
        other_output = critic(obs, other_changed)

    assert torch.allclose(own_output.all_q[:, 0], output.all_q[:, 0], atol=1e-6)
    assert not torch.allclose(other_output.all_q[:, 0], output.all_q[:, 0])
    assert output.attention is not None
    assert output.attention_regularization.item() > 0.0


def test_maac_attention_regularizes_released_unscaled_logits() -> None:
    torch.manual_seed(4)
    critic = AttentionCritic(
        n_agents=3, obs_dim=4, action_dim=3, hidden_dim=8, attend_heads=2,
    )
    critic.eval()
    obs = torch.randn(5, 3, 4)
    actions = F.one_hot(torch.randint(0, 3, (5, 3)), 3).float()

    with torch.no_grad():
        output = critic(obs, actions)
        encoded = [
            encoder(torch.cat([obs[:, agent_id], actions[:, agent_id]], dim=-1))
            for agent_id, encoder in enumerate(critic.state_action_encoders)
        ]
        states = [
            encoder(obs[:, agent_id])
            for agent_id, encoder in enumerate(critic.state_encoders)
        ]
        expected = obs.new_zeros(())
        for head_id in range(critic.attend_heads):
            keys = [critic.key_extractors[head_id](value) for value in encoded]
            selectors = [
                critic.selector_extractors[head_id](value) for value in states
            ]
            for agent_id in range(critic.n_agents):
                other_keys = torch.stack(
                    [key for other_id, key in enumerate(keys) if other_id != agent_id],
                    dim=1,
                )
                raw_logits = torch.matmul(
                    selectors[agent_id].unsqueeze(1), other_keys.transpose(1, 2),
                )
                expected = expected + 1e-3 * raw_logits.square().mean()

    assert torch.allclose(output.attention_regularization, expected, atol=1e-8)


def test_maac_config_matches_released_experiment() -> None:
    config = MAACConfig()
    assert config.entropy_temperature == pytest.approx(0.01)
    assert config.gamma == pytest.approx(0.99)
    assert config.tau == pytest.approx(0.001)
    assert config.batch_size == 1024
    assert config.replay_capacity == 1_000_000
    assert config.update_interval == 100
    assert config.updates_per_interval == 4


def test_maac_paper_task_exposes_each_agents_reward() -> None:
    environment = make_env("paper_navigation", 3, 25, 5)
    environment.reset(seed=5)
    _, reward, _, _, info = environment.step(np.zeros(3, dtype=np.int64))
    agent_rewards = info["agent_rewards"]
    assert agent_rewards.shape == (3,)
    assert reward == pytest.approx(float(agent_rewards.mean()))


def test_maac_trainer_keeps_paper_reward_vector(monkeypatch) -> None:
    captured: list[tuple[int, ...]] = []
    original = MAACLearner.store_transition

    def record(self, obs, actions, rewards, next_obs, dones):
        captured.append(np.asarray(rewards).shape)
        return original(self, obs, actions, rewards, next_obs, dones)

    monkeypatch.setattr(MAACLearner, "store_transition", record)
    monkeypatch.setattr(
        train_maac_module,
        "_evaluate",
        lambda *args, **kwargs: {"returns": [], "successes": []},
    )
    train_maac_module.train(
        env="paper_navigation", episodes=1, evaluation_episodes=1,
    )
    assert captured == [(3,)] * 25


def test_maac_soft_target_matches_paper_equation_four(monkeypatch) -> None:
    torch.manual_seed(0)
    n, obs_dim, action_dim, batch_size = 2, 3, 2, 3
    learner = MAACLearner(
        n, obs_dim, action_dim, hidden_dim=8, attend_heads=2,
        config=MAACConfig(batch_size=batch_size),
    )
    learner.train()
    rewards = torch.tensor([[1.0, 1.5], [-0.5, 0.25], [2.0, -1.0]])
    dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    batch = ReplayBatch(
        obs=torch.randn(batch_size, n, obs_dim),
        actions=torch.randint(0, action_dim, (batch_size, n)),
        rewards=rewards,
        next_obs=torch.randn(batch_size, n, obs_dim),
        dones=dones,
    )

    q_const = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    learner.target_critic.forward = lambda obs, actions: MAACCriticOutput(
        q_taken=q_const.clone(),
        all_q=torch.empty(0),
        attention=None,
        attention_regularization=torch.tensor(0.0),
    )
    log_pi_const = [-1.0, -2.0]
    for agent_id, agent in enumerate(learner.agents):
        def stub_sample(obs, _log_pi=log_pi_const[agent_id], **kwargs):
            rows = obs.shape[0]
            one_hot = F.one_hot(torch.zeros(rows, dtype=torch.long), action_dim).float()
            filler = torch.zeros(rows, action_dim)
            return (
                one_hot,
                torch.zeros(rows, dtype=torch.long),
                filler,
                filler,
                filler,
                torch.full((rows,), _log_pi),
                torch.zeros(rows),
            )

        agent.target_actor.sample = stub_sample

    captured: list[torch.Tensor] = []
    original_mse = F.mse_loss

    def recording_mse_loss(prediction, target, *args, **kwargs):
        captured.append(target.detach().clone())
        return original_mse(prediction, target, *args, **kwargs)

    monkeypatch.setattr(maac_module.F, "mse_loss", recording_mse_loss)
    learner._update_batch(batch)

    log_pi = torch.tensor(log_pi_const).unsqueeze(0).expand(batch_size, n)
    expected = rewards + learner.config.gamma * (1.0 - dones) * (
        q_const - learner.config.entropy_temperature * log_pi
    )
    assert len(captured) == n
    assert torch.allclose(torch.stack(captured, dim=1), expected, atol=1e-6)


def test_maac_scales_only_shared_critic_gradients() -> None:
    critic = AttentionCritic(3, 4, 2, hidden_dim=8, attend_heads=2)
    for parameter in critic.parameters():
        parameter.grad = torch.ones_like(parameter)
    critic.scale_shared_grads()
    assert all(
        torch.allclose(parameter.grad, torch.full_like(parameter, 1 / 3))
        for parameter in critic.shared_parameters()
    )
    assert all(
        torch.equal(parameter.grad, torch.ones_like(parameter))
        for module in (critic.state_encoders, critic.agent_q_heads)
        for parameter in module.parameters()
    )


def test_maac_runs_four_updates_every_100_steps(monkeypatch) -> None:
    learner = MAACLearner(
        2, 3, 2, hidden_dim=8, attend_heads=2,
        config=MAACConfig(batch_size=2, replay_capacity=128),
    )
    transition = (
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    marker = MAACUpdate(1.0, 2.0, 3.0)
    calls = 0

    def fake_update(batch):
        nonlocal calls
        calls += 1
        return marker

    monkeypatch.setattr(learner, "_update_batch", fake_update)
    for _ in range(99):
        learner.store_transition(*transition)
        assert learner.update() == []
    learner.store_transition(*transition)
    assert learner.update() == [marker] * 4
    assert calls == 4


def test_maac_target_networks_are_owned_and_gradient_free() -> None:
    learner = MAACLearner(2, 3, 2, hidden_dim=8, attend_heads=2)
    assert learner.target_critic is not learner.critic
    assert all(not parameter.requires_grad for parameter in learner.target_critic.parameters())
    for agent in learner.agents:
        assert agent.target_actor is not agent.actor
        assert all(not parameter.requires_grad for parameter in agent.target_actor.parameters())


def test_maac_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "maac.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        hidden_dim=32,
        attend_heads=4,
        buffer_size=64,
        batch_size=4,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "maac"
    assert summary["env"] == "navigation"
    assert summary["config"]["entropy_temperature"] == pytest.approx(0.01)
    assert summary["config"]["horizon"] == 6
    assert checkpoint.exists()
