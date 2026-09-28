from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_i2c import train
from marl_envs import make_env
from modmarl.algorithms.i2c import I2CAgent, I2CPolicy, I2CReplayBatch, I2CReplayBuffer


def test_i2c_policy_shapes() -> None:
    policy = I2CPolicy(n_agents=3, obs_dim=5, action_dim=4, message_dim=16, hidden_dim=32)
    obs = torch.randn(6, 3, 5)
    logits, prior_logits = policy(obs)
    assert tuple(logits.shape) == (6, 3, 4)
    assert tuple(prior_logits.shape) == (6, 3, 3)
    one_hot, action_idx, sampled_logits, priors = policy.sample(obs, hard=True)
    assert tuple(one_hot.shape) == (6, 3, 4)
    assert tuple(action_idx.shape) == (6, 3)
    assert tuple(sampled_logits.shape) == (6, 3, 4)
    assert tuple(priors.shape) == (6, 3, 3)


def test_i2c_prior_network_shapes() -> None:
    policy = I2CPolicy(n_agents=4, obs_dim=7, action_dim=5, message_dim=16, hidden_dim=32)
    obs = torch.randn(6, 4, 7)

    # Prior b_i(o_i, id_j) scores every (receiver, sender) pair.
    prior_logits = policy.prior(obs, torch.randn(6, 4, 4, 2))
    assert tuple(prior_logits.shape) == (6, 4, 4)

    # Raw observations are the transmitted payloads; the recurrent encoder runs after gating.
    encodings = policy.encode(obs)
    assert tuple(encodings.shape) == (6, 4, 7)

    # The communication gate is a hard {0, 1} mask with a zero diagonal (no self-communication).
    gate = policy._gate(prior_logits)
    assert tuple(gate.shape) == (6, 4, 4)
    assert torch.all((gate == 0.0) | (gate == 1.0))
    assert torch.all(torch.diagonal(gate, dim1=-2, dim2=-1) == 0.0)


def test_i2c_uses_release_parameter_sharing() -> None:
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    assert hasattr(agent.policy, "actor_head")
    assert hasattr(agent.policy, "message_encoder")
    assert hasattr(agent.policy, "prior_net")
    assert hasattr(agent, "critic")
    larger = I2CPolicy(n_agents=7, obs_dim=5, action_dim=4, hidden_dim=16)
    assert sum(parameter.numel() for parameter in agent.policy.parameters()) == sum(
        parameter.numel() for parameter in larger.parameters()
    )


def test_i2c_prior_depends_on_candidate_relative_position() -> None:
    torch.manual_seed(9)
    policy = I2CPolicy(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(2, 3, 5)
    locations = torch.zeros(2, 3, 3, 2)
    baseline = policy.prior(obs, locations)
    locations[:, 0, 1] = torch.tensor([3.0, -2.0])
    changed = policy.prior(obs, locations)
    assert not torch.allclose(baseline[:, 0, 1], changed[:, 0, 1])
    torch.testing.assert_close(baseline[:, 1:], changed[:, 1:])


def test_i2c_critic_shapes() -> None:
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, message_dim=16, hidden_dim=32)
    obs = torch.randn(6, 3, 5)
    actions = torch.zeros(6, 3, 4)
    actions[:, :, 0] = 1.0
    assert tuple(agent.critic(obs, actions).shape) == (6,)


def test_i2c_actor_deterministic_sample_matches_argmax() -> None:
    policy = I2CPolicy(n_agents=3, obs_dim=5, action_dim=4, message_dim=16, hidden_dim=32)
    obs = torch.randn(6, 3, 5)
    logits, _ = policy(obs)
    one_hot, action_idx, _, _ = policy.sample(obs, deterministic=True)
    expected_idx = logits.argmax(dim=-1)
    assert torch.equal(action_idx, expected_idx)
    assert torch.equal(one_hot, torch.nn.functional.one_hot(expected_idx, num_classes=4).to(logits.dtype))


def test_i2c_prior_threshold_gates_messages_end_to_end() -> None:
    torch.manual_seed(0)
    # sigmoid(logit) is always in (0, 1): threshold 0.0 opens every gate, 1.1 opens none.
    policy = I2CPolicy(n_agents=3, obs_dim=5, action_dim=4, message_dim=16, hidden_dim=32, threshold=0.0)
    obs = torch.randn(4, 3, 5)
    open_logits, _ = policy(obs)

    policy.threshold = 1.1
    closed_logits, _ = policy(obs)

    # Messages actually reach the actor: gating them off changes the action logits.
    assert not torch.allclose(open_logits, closed_logits)
    # With every gate closed, every recurrent input slot is zero (the LSTM may
    # still produce a learned bias/state response to that explicit no-message sequence).
    zero_gate = torch.zeros(4, 3, 3)
    assert torch.allclose(closed_logits, policy.act(obs, policy.aggregate(obs, zero_gate)))


def test_i2c_message_encoder_never_sees_unrequested_observations() -> None:
    torch.manual_seed(4)
    policy = I2CPolicy(n_agents=3, obs_dim=5, action_dim=4, message_dim=8, hidden_dim=16)
    obs = torch.randn(2, 3, 5)
    gate = torch.zeros(2, 3, 3)
    gate[:, 0, 1] = 1.0
    baseline = policy.aggregate(obs, gate)

    hidden_sender_changed = obs.clone()
    hidden_sender_changed[:, 2] += 100.0
    torch.testing.assert_close(policy.aggregate(hidden_sender_changed, gate)[:, 0], baseline[:, 0])

    requested_sender_changed = obs.clone()
    requested_sender_changed[:, 1] += 100.0
    assert not torch.allclose(policy.aggregate(requested_sender_changed, gate)[:, 0], baseline[:, 0])


def test_i2c_message_encoder_uses_three_release_slots() -> None:
    policy = I2CPolicy(n_agents=7, obs_dim=5, action_dim=4, hidden_dim=16, max_messages=3)
    observed_shapes = []
    handle = policy.message_encoder.register_forward_pre_hook(
        lambda _module, inputs: observed_shapes.append(tuple(inputs[0].shape)),
    )
    try:
        policy.aggregate(torch.randn(2, 7, 5), torch.ones(2, 7, 7) - torch.eye(7))
    finally:
        handle.remove()
    assert observed_shapes == [(14, 3, 5)]


def test_i2c_candidate_mask_blocks_nonvisible_teammates() -> None:
    policy = I2CPolicy(n_agents=4, obs_dim=5, action_dim=4, hidden_dim=16, threshold=0.0)
    prior_logits = torch.ones(2, 4, 4)
    candidate_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    candidate_mask[:, 0, 2] = True
    gate = policy._gate(prior_logits, candidate_mask)
    assert gate.sum() == 2
    assert torch.all(gate[:, 0, 2] == 1)


def test_i2c_causal_influence_uses_jointly_normalized_marginal() -> None:
    num_actions = 4
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=num_actions, message_dim=3, hidden_dim=32)
    obs = torch.randn(2, 3, 5)

    class _StubCritic(torch.nn.Module):
        # Agents 0 and 1 have a strong action interaction; agent 2 is independent.
        # Marginalizing either interacting agent changes the other's conditional action
        # distribution, so the two directed pairs exceed the KL margin.
        def forward(self, obs_in: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return 3.0 * (
                actions[:, 0, 0] * actions[:, 1, 0]
                + actions[:, 0, 1] * actions[:, 1, 1]
            )

    agent.critic = _StubCritic()
    actions = torch.nn.functional.one_hot(torch.zeros(2, 3, dtype=torch.long), num_actions).float()
    influence = agent.causal_influence(obs, actions)
    conditional = torch.softmax(torch.tensor([3.0, 0.0, 0.0, 0.0]), dim=0)
    joint = torch.zeros(4, 4)
    joint[0, 0] = joint[1, 1] = 3.0
    marginal = torch.softmax(joint.flatten(), dim=0).view(4, 4).sum(dim=1)
    expected = (conditional * (conditional.log() - marginal.log())).sum()
    torch.testing.assert_close(influence[:, 0, 1], expected.expand(2))
    torch.testing.assert_close(influence[:, 1, 0], expected.expand(2))
    assert torch.allclose(influence[:, 2], torch.zeros_like(influence[:, 2]))


def test_i2c_correlation_target_matches_released_normalized_boltzmann() -> None:
    torch.manual_seed(13)
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(6, 3, 5)
    actions = torch.softmax(torch.randn(6, 3, 4), dim=-1)
    actual = agent.correlation_action_distribution(obs, actions, receiver=1)

    candidates = actions[:, None].expand(6, 4, 3, 4).clone()
    candidates[:, :, 1] = torch.eye(4)
    tiled_obs = obs[:, None].expand(-1, 4, -1, -1)
    q = agent.critic(tiled_obs.reshape(-1, 3, 5), candidates.reshape(-1, 3, 4)).view(6, 4)
    centered = q - q.mean(dim=-1, keepdim=True)
    scale = centered.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    expected = torch.softmax(8.0 * centered / scale, dim=-1)
    torch.testing.assert_close(actual, expected)


def test_i2c_gradient_separation_between_policy_and_prior_losses() -> None:
    torch.manual_seed(0)
    obs = torch.randn(4, 3, 5)

    # Policy loss must not touch the prior parameters (the gate is detached).
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, message_dim=16, hidden_dim=32)
    actions, _, logits, _ = agent.policy.sample(obs, hard=True)
    policy_loss = -agent.critic(obs, actions).mean() + 1e-3 * logits.pow(2).mean()
    policy_loss.backward()
    assert all(p.grad is None for p in agent.policy.prior_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.actor_parameters())

    # Prior supervision must not touch actor parameters.
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, message_dim=16, hidden_dim=32)
    labels = torch.zeros(4, 3, 3)
    mask = ~torch.eye(3, dtype=torch.bool)
    prior_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        agent.policy.prior(obs, torch.randn(4, 3, 3, 2))[:, mask], labels[:, mask],
    )
    prior_loss.backward()
    assert all(p.grad is None for p in agent.policy.actor_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.prior_parameters())


def test_i2c_agent_soft_update_runs() -> None:
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, message_dim=16, hidden_dim=32)
    agent.soft_update(0.01)


def test_i2c_actor_critic_update_does_not_refit_frozen_prior() -> None:
    torch.manual_seed(7)
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, message_dim=8, hidden_dim=16)
    assert agent.influence_temperature == 1.0
    assert agent.correlation_temperature == 8.0
    assert agent.influence_percentile == 80.0
    assert all(not parameter.requires_grad for parameter in agent.target_policy.parameters())
    batch = I2CReplayBatch(
        obs=torch.randn(6, 3, 5), actions=torch.softmax(torch.randn(6, 3, 4), dim=-1),
        rewards=torch.randn(6), next_obs=torch.randn(6, 3, 5), dones=torch.zeros(6),
    )
    prior_before = [parameter.detach().clone() for parameter in agent.policy.prior_parameters()]
    metrics = agent.update(batch, receiver=0)
    assert metrics.critic_loss >= 0.0
    assert metrics.correlation_loss >= 0.0
    assert 0.0 <= metrics.communication_rate <= 1.0
    assert all(
        torch.equal(before, after)
        for before, after in zip(prior_before, agent.policy.prior_parameters())
    )


def test_i2c_load_frozen_prior_reinitializes_only_phase_two_networks() -> None:
    torch.manual_seed(41)
    teacher = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    final = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    actor_before = [parameter.detach().clone() for parameter in final.policy.actor_parameters()]
    final.load_frozen_prior(teacher)
    for expected, actual in zip(teacher.policy.prior_parameters(), final.policy.prior_parameters()):
        torch.testing.assert_close(actual, expected)
        assert not actual.requires_grad
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_before, final.policy.actor_parameters())
    )


def test_i2c_single_agent_update_has_no_prior_pairs() -> None:
    agent = I2CAgent(n_agents=1, obs_dim=3, action_dim=2, message_dim=4, hidden_dim=8)
    batch = I2CReplayBatch(
        obs=torch.randn(3, 1, 3), actions=torch.softmax(torch.randn(3, 1, 2), dim=-1),
        rewards=torch.randn(3), next_obs=torch.randn(3, 1, 3), dones=torch.zeros(3),
    )
    metrics = agent.update(batch, receiver=0)
    assert metrics.communication_rate == 0.0


def test_i2c_selected_receiver_update_changes_shared_network() -> None:
    torch.manual_seed(21)
    agent = I2CAgent(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    batch = I2CReplayBatch(
        obs=torch.randn(8, 3, 5), actions=torch.softmax(torch.randn(8, 3, 4), dim=-1),
        rewards=torch.randn(8), next_obs=torch.randn(8, 3, 5), dones=torch.zeros(8),
    )
    before = [parameter.detach().clone() for parameter in agent.policy.actor_head.parameters()]
    agent.update(batch, receiver=0)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy.actor_head.parameters())
    )


def test_i2c_replay_preserves_soft_gumbel_actions() -> None:
    replay = I2CReplayBuffer(capacity=4, n_agents=2, obs_dim=3, action_dim=3)
    action = torch.tensor([[0.15, 0.70, 0.15], [0.20, 0.25, 0.55]]).numpy()
    messages = torch.randn(2, 3, 3).numpy()
    replay.add(
        obs=torch.zeros(2, 3).numpy(), actions=action, reward=1.0,
        next_obs=torch.ones(2, 3).numpy(), done=False, messages=messages,
    )
    batch = replay.sample(1, torch.device("cpu"))
    torch.testing.assert_close(batch.actions[0], torch.as_tensor(action))
    torch.testing.assert_close(batch.messages[0], torch.as_tensor(messages))
    assert not torch.all((batch.actions == 0.0) | (batch.actions == 1.0))


def test_i2c_replayed_messages_bypass_current_prior() -> None:
    torch.manual_seed(31)
    policy = I2CPolicy(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16)
    obs = torch.randn(2, 3, 5)
    messages = torch.randn(2, 3, 3, 5)
    logits = policy.act_from_messages(obs, messages)
    with torch.no_grad():
        for parameter in policy.prior_parameters():
            parameter.add_(100.0 * torch.randn_like(parameter))
    torch.testing.assert_close(policy.act_from_messages(obs, messages), logits)


def test_i2c_paper_environment_matches_published_candidate_contract() -> None:
    environment = make_env("paper_i2c_navigation", n_agents=99, horizon=99, seed=3)
    obs, _ = environment.reset(seed=3)
    locations, mask = environment.communication_candidates()
    assert obs.shape == (7, 16)
    assert environment.horizon == 40
    assert np.all(mask.sum(axis=1) == 3)
    positions = np.stack([agent.state.p_pos for agent in environment._env.world.agents])
    np.testing.assert_allclose(locations, positions[:, None] - positions[None, :], atol=1e-6)


def test_i2c_paper_environment_matches_author_source_golden_step() -> None:
    environment = make_env("paper_i2c_navigation", n_agents=7, horizon=40, seed=3)
    obs, _ = environment.reset(seed=3)
    expected_first_obs = np.array([
        0.0, 0.0, 0.101595805, 0.416295645, -0.012297769, 0.144333884,
        0.250913999, -0.234570010, -0.534545642, -0.030019809,
        -0.519786327, -0.394640435, 0.684298104, 0.376290533,
        0.196692290, -0.859321080,
    ])
    np.testing.assert_allclose(obs[0], expected_first_obs, atol=1e-6)
    next_obs, reward, _, _, info = environment.step(np.full((7, 5), 0.2))
    np.testing.assert_allclose(next_obs[0], expected_first_obs, atol=1e-6)
    assert reward == pytest.approx(-2.0790471416175804)
    assert info["benchmark_reward"] == pytest.approx(-2.0790471416175804)


def test_i2c_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "i2c.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        message_dim=16,
        hidden_dim=32,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "i2c"
    assert summary["env"] == "navigation"
    assert summary["n_agents"] == 3
    assert "final_return" in summary
    assert "best_return" in summary
    assert summary["message_ablated_evaluation"]["communication_rates"] == [0.0] * 32
    assert 0.0 <= summary["communication_rate"] <= 1.0
    assert checkpoint.exists()


def test_i2c_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = tmp_path / "i2c.pt"
    train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        message_dim=16,
        hidden_dim=32,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        checkpoint=str(checkpoint),
    )
    state = torch.load(str(checkpoint), weights_only=True)
    assert set(state) == {
        "model", "policy_optimizer", "critic_optimizer", "prior_state",
        "influence_threshold",
    }
