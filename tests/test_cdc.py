from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_cdc import _evaluate, _sample_without_replacement, train
from marl_envs import (
    FormationControlEnv,
    LeaderFollowerTargetEnv,
    LineControlEnv,
    NavigationControlEnv,
    PaperParticleEnv,
    PettingZooSimpleSpreadEnv,
    SimpleSpreadMPEEnv,
    TargetSignalingEnv,
    paper_particle_env_available,
    pettingzoo_simple_spread_available,
)
from modmarl import (
    AttentionCritic,
    CDCAgent,
    CDCConfig,
    MAACAgent,
    MADDPGAgent,
    PaperEquationCDCPolicy,
    ReleasedCDCCritic,
    ReleasedCDCPolicy,
    ReplayBuffer,
)


class CDCSmokeTests(unittest.TestCase):
    def test_released_policy_matches_independent_release_equations(self) -> None:
        torch.manual_seed(4)
        policy = ReleasedCDCPolicy(obs_dim=3, action_dim=2, message_dim=5).eval()
        obs = torch.randn(2, 3, 3)
        messages, diagnostics = policy.compute_messages(obs)

        pair_messages = torch.empty(2, 3, 3, 5)
        adjacency = torch.empty(2, 3, 3)
        with torch.no_grad():
            edge = policy.edge_network
            for sender in range(3):
                for receiver in range(3):
                    pair = torch.cat([obs[:, sender], obs[:, receiver]], dim=-1)
                    pair = torch.nn.functional.batch_norm(
                        pair, edge.input_norm.running_mean, edge.input_norm.running_var,
                        edge.input_norm.weight, edge.input_norm.bias, training=False,
                        eps=edge.input_norm.eps,
                    )
                    message = edge.fc2(torch.relu(edge.fc1(pair)))
                    pair_messages[:, sender, receiver] = message
                    adjacency[:, sender, receiver] = torch.sigmoid(edge.fc3(message)).squeeze(-1)
            degree = adjacency.sum(2)
            laplacian = torch.diag_embed(degree) - adjacency
            inv_sqrt = torch.diag_embed(degree.rsqrt())
            normalised = inv_sqrt @ laplacian @ inv_sqrt
            selected = torch.zeros_like(adjacency)
            previous = torch.exp(-0.05 * normalised)
            for diffusion_time in torch.arange(0.1, 15.0, 0.05):
                current = torch.exp(-diffusion_time * normalised)
                relative = ((current - previous) / previous).abs()
                candidate = torch.where(relative < 0.05, relative, torch.zeros_like(relative))
                selected = torch.where(selected == 0, candidate, selected)
                previous = current
                if torch.all(selected != 0):
                    break
            expected_messages = (selected.unsqueeze(-1) * pair_messages).sum(1)

        self.assertTrue(torch.allclose(diagnostics["adjacency"], adjacency, atol=1e-7))
        self.assertTrue(torch.allclose(diagnostics["heat_weights"], selected, atol=1e-7))
        self.assertTrue(torch.allclose(messages, expected_messages, atol=1e-7))

    def test_released_policy_keeps_directed_self_edges(self) -> None:
        policy = ReleasedCDCPolicy(obs_dim=4, action_dim=3, message_dim=8).eval()
        _, diagnostics = policy(torch.randn(2, 3, 4))
        adjacency = diagnostics["adjacency"]
        self.assertTrue(torch.all(torch.diagonal(adjacency, dim1=-2, dim2=-1) > 0))
        self.assertFalse(torch.allclose(adjacency, adjacency.transpose(1, 2)))

    def test_paper_equation_variant_enforces_published_symmetry(self) -> None:
        released = ReleasedCDCPolicy(obs_dim=4, action_dim=3, message_dim=8).eval()
        clean = PaperEquationCDCPolicy(obs_dim=4, action_dim=3, message_dim=8).eval()
        released_parameters = released.state_dict()
        del released_parameters["diffusion_times"]
        clean.load_state_dict(released_parameters, strict=False)
        obs = torch.randn(2, 3, 4)
        _, released_info = released(obs)
        _, clean_info = clean(obs)
        self.assertTrue(torch.allclose(clean_info["adjacency"], clean_info["adjacency"].transpose(1, 2)))
        self.assertTrue(torch.allclose(clean_info["pair_messages"], clean_info["pair_messages"].transpose(1, 2)))
        self.assertFalse(torch.equal(released_info["adjacency"], clean_info["adjacency"]))
        self.assertFalse(torch.allclose(released_info["heat_weights"], clean_info["heat_weights"]))

    def test_complete_agent_updates_online_but_not_target_parameters_directly(self) -> None:
        torch.manual_seed(8)
        agent = CDCAgent(obs_dim=4, action_dim=3, hidden_dim=8, message_dim=8)
        actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        target_before = [parameter.detach().clone() for parameter in agent.target_actor.parameters()]
        update = agent.update(
            torch.randn(12, 3, 4),
            torch.randint(0, 3, (12, 3)),
            torch.randn(12),
            torch.randn(12, 3, 4),
            torch.zeros(12),
        )
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
        self.assertTrue(all(parameter.grad is None for parameter in agent.target_actor.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(target_before, agent.target_actor.parameters())))
        self.assertTrue(np.isfinite(update.actor_loss))
        self.assertTrue(np.isfinite(update.critic_loss))

    def test_reference_schedule_updates_every_hundred_steps(self) -> None:
        schedule = CDCConfig()
        self.assertFalse(schedule.update_due(environment_steps=100, replay_size=1023))
        self.assertFalse(schedule.update_due(environment_steps=101, replay_size=1024))
        self.assertTrue(schedule.update_due(environment_steps=100, replay_size=1024))

    def test_no_message_ablation_removes_all_observation_dependence(self) -> None:
        agent = CDCAgent(obs_dim=4, action_dim=3, hidden_dim=8, message_dim=8)
        first = agent.act_without_messages(torch.randn(2, 3, 4))
        second = agent.act_without_messages(torch.randn(2, 3, 4) * 100.0)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[:, :1].expand_as(first), first))

    def test_evaluation_does_not_change_training_random_streams(self) -> None:
        agent = CDCAgent(obs_dim=14, action_dim=5, hidden_dim=8, message_dim=8)
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state()
        _evaluate(agent, "navigation", 3, 2, 91, 1)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_paper_equation_batched_grid_matches_serial_equations(self) -> None:
        policy = PaperEquationCDCPolicy(obs_dim=3, action_dim=2, message_dim=4).eval()
        self.assertEqual(policy.diffusion_times.numel(), 300)
        laplacian = torch.tensor(
            [[[0.7, -0.2, -0.5], [-0.1, 0.6, -0.5], [-0.3, -0.2, 0.5]]],
            dtype=torch.float32,
        )
        batched = policy._diffusion_weights(laplacian)
        serial = torch.zeros_like(laplacian)
        previous = torch.matrix_exp(-policy.diffusion_times[0] * laplacian)
        for diffusion_time in policy.diffusion_times[1:]:
            current = torch.matrix_exp(-diffusion_time * laplacian)
            relative = ((current - previous) / previous.clamp_min(1e-8)).abs()
            # Equation (5) anchors the test on p and evaluates H at p_hat = p, so the
            # retained value is `previous`, not the next grid point.
            serial = torch.where((serial == 0) & (relative < 0.05), previous, serial)
            previous = current
        self.assertTrue(torch.allclose(batched, serial, atol=1e-6, rtol=1e-6))

    def test_released_critic_shape_and_gradients(self) -> None:
        critic = ReleasedCDCCritic(obs_dim=4, action_dim=3, hidden_dim=8)
        obs = torch.randn(5, 3, 4)
        actions = torch.nn.functional.one_hot(torch.randint(0, 3, (5, 3)), 3).float()
        values = critic(obs, actions)
        values.mean().backward()
        self.assertEqual(tuple(values.shape), (5,))
        self.assertTrue(all(parameter.grad is not None for parameter in critic.parameters()))

    def test_environment_shapes(self) -> None:
        env = LeaderFollowerTargetEnv(n_agents=3, horizon=8, seed=3)
        obs, _ = env.reset(seed=3)
        self.assertEqual(obs.shape, (3, env.obs_dim))
        next_obs, reward, terminated, truncated, info = env.step([0, 1, 2])
        self.assertEqual(next_obs.shape, (3, env.obs_dim))
        self.assertIsInstance(reward, float)
        self.assertIn("mean_distance", info)
        self.assertFalse(terminated and truncated)

    def test_signaling_env_shapes(self) -> None:
        env = TargetSignalingEnv(n_agents=3, seed=5)
        obs, _ = env.reset(seed=5)
        self.assertEqual(obs.shape, (3, env.obs_dim))
        next_obs, reward, terminated, truncated, info = env.step([0, 0, 0])
        self.assertEqual(next_obs.shape, (3, env.obs_dim))
        self.assertEqual(env.num_actions, 2)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn("target", info)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)

    def test_mpe_like_env_shapes(self) -> None:
        for env in (
            NavigationControlEnv(n_agents=3, horizon=5, seed=11),
            FormationControlEnv(n_agents=4, horizon=5, seed=13),
            LineControlEnv(n_agents=4, horizon=5, seed=17),
            SimpleSpreadMPEEnv(n_agents=3, horizon=5, seed=19),
        ):
            obs, _ = env.reset(seed=env.n_agents)
            self.assertEqual(obs.shape[0], env.n_agents)
            next_obs, reward, terminated, truncated, info = env.step([0] * env.n_agents)
            self.assertEqual(next_obs.shape, obs.shape)
            self.assertIsInstance(reward, float)
            self.assertIn("mean_distance", info)
            self.assertIn("collisions", info)
            self.assertFalse(terminated and truncated)

    def test_pettingzoo_simple_spread_wrapper_shapes(self) -> None:
        if not pettingzoo_simple_spread_available():
            self.skipTest("PettingZoo simple_spread dependencies are not installed")
        env = PettingZooSimpleSpreadEnv(n_agents=3, horizon=5, seed=23)
        obs, info = env.reset(seed=23)
        self.assertEqual(obs.shape, (3, env.obs_dim))
        self.assertIn("mean_distance", info)
        next_obs, reward, terminated, truncated, step_info = env.step([0, 1, 2])
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(reward, float)
        self.assertIn("mean_distance", step_info)
        self.assertFalse(terminated and truncated)
        env.close()

    def test_paper_particle_wrapper_shapes(self) -> None:
        if not paper_particle_env_available():
            self.skipTest("Archived particle env dependencies are not available")
        env = PaperParticleEnv(env_name="paper_navigation", horizon=5, seed=23)
        obs, info = env.reset(seed=23)
        self.assertEqual(obs.shape, (env.n_agents, env.obs_dim))
        self.assertIn("mean_distance", info)
        next_obs, reward, terminated, _truncated, step_info = env.step([0] * env.n_agents)
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(reward, float)
        self.assertIn("collisions", step_info)
        self.assertFalse(terminated)
        env.close()

    def test_maddpg_agent_shapes(self) -> None:
        agent = MADDPGAgent(n_agents=3, obs_dim=5, action_dim=5, hidden_dim=32)
        obs = torch.randn(4, 5)
        global_obs = torch.randn(4, 3, 5)
        one_hot, action_idx, _ = agent.actor.sample(obs, hard=True, deterministic=False)
        q_values = agent.critic(global_obs, torch.randn(4, 3, 5))
        self.assertEqual(tuple(one_hot.shape), (4, 5))
        self.assertEqual(tuple(action_idx.shape), (4,))
        self.assertEqual(tuple(q_values.shape), (4,))

    def test_maac_attention_shapes(self) -> None:
        agent = MAACAgent(obs_dim=5, action_dim=5, hidden_dim=32)
        critic = AttentionCritic(n_agents=3, obs_dim=5, action_dim=5, hidden_dim=32, attend_heads=4)
        obs = torch.randn(4, 3, 5)
        actions = torch.nn.functional.one_hot(torch.randint(0, 5, (4, 3)), num_classes=5).to(dtype=torch.float32)
        one_hot, action_idx, logits, probs, log_probs, chosen_log_prob, entropy = agent.actor.sample(
            obs[:, 0],
            deterministic=False,
        )
        critic_output = critic(obs, actions, return_attention=True)
        self.assertEqual(tuple(one_hot.shape), (4, 5))
        self.assertEqual(tuple(action_idx.shape), (4,))
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(tuple(probs.shape), (4, 5))
        self.assertEqual(tuple(log_probs.shape), (4, 5))
        self.assertEqual(tuple(chosen_log_prob.shape), (4,))
        self.assertEqual(tuple(entropy.shape), (4,))
        self.assertEqual(tuple(critic_output.q_taken.shape), (4, 3))
        self.assertEqual(tuple(critic_output.all_q.shape), (4, 3, 5))
        self.assertEqual(len(critic_output.attention), 3)

    def test_replay_buffer_sample_shapes(self) -> None:
        buffer = ReplayBuffer(capacity=32, n_agents=3, obs_dim=5)
        obs = torch.randn(3, 5).numpy()
        next_obs = torch.randn(3, 5).numpy()
        for _ in range(10):
            buffer.add(obs=obs, actions=np.array([0, 1, 2]), reward=1.0, next_obs=next_obs, done=False)
        batch = buffer.sample(batch_size=4, device=torch.device("cpu"))
        self.assertEqual(tuple(batch.obs.shape), (4, 3, 5))
        self.assertEqual(tuple(batch.actions.shape), (4, 3))
        self.assertEqual(tuple(batch.rewards.shape), (4,))

    def test_released_replay_sample_has_unique_transitions(self) -> None:
        buffer = ReplayBuffer(capacity=16, n_agents=3, obs_dim=2)
        for index in range(8):
            obs = np.full((3, 2), index, dtype=np.float32)
            buffer.add(obs=obs, actions=np.zeros(3), reward=float(index), next_obs=obs, done=False)
        batch = _sample_without_replacement(buffer, batch_size=8, device=torch.device("cpu"))
        self.assertEqual(len(torch.unique(batch.rewards)), 8)

    def test_released_rollout_uses_policy_during_replay_warmup(self) -> None:
        with patch("examples.train_cdc.np.random.randint", side_effect=AssertionError("random warmup action")):
            result = train(
                env="navigation", n_agents=3, horizon=2, episodes=1, seed=3,
                hidden_dim=8, message_dim=8, batch_size=32, warmup_steps=32,
                variant="released", evaluation_episodes=1,
            )
        self.assertEqual(len(result["returns"]), 1)

    def test_released_training_seeds_environment_only_once(self) -> None:
        class RecordingEnv:
            n_agents = 3
            obs_dim = 4
            num_actions = 3
            horizon = 1

            def __init__(self) -> None:
                self.reset_seeds: list[int | None] = []

            def reset(self, *, seed=None):
                self.reset_seeds.append(seed)
                return np.zeros((3, 4), dtype=np.float32), {}

            def step(self, action):
                return np.zeros((3, 4), dtype=np.float32), -1.0, False, True, {}

        environment = RecordingEnv()
        with patch("examples.train_cdc.make_env", return_value=environment):
            train(
                episodes=3, hidden_dim=8, message_dim=8, batch_size=32,
                warmup_steps=32, variant="released", evaluation_episodes=0,
            )
        self.assertEqual(environment.reset_seeds, [None, None, None])

if __name__ == "__main__":
    unittest.main()


def test_cdc_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "cdc.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        hidden_dim=32,
        message_dim=16,
        diffusion_steps=6,
        diffusion_max=2.0,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        update_every=4,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "cdc"
    assert summary["env"] == "navigation"
    assert summary["updates"] == 3
    assert len(summary["random_evaluation"]["returns"]) == 2
    assert summary["communication_rate"] == 1.0
    assert summary["config"]["diffusion_steps"] == 6
    assert checkpoint.exists()
    saved = torch.load(checkpoint, weights_only=True)
    assert {
        "agent", "actor_optimizer", "critic_optimizer", "environment_steps", "updates", "variant",
    } <= saved.keys()


def test_cdc_train_rejects_invalid_update_cadence() -> None:
    with pytest.raises(ValueError, match="update_every"):
        train(episodes=0, update_every=0)
