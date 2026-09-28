"""Golden tests for TarMAC's targeted messages and paper actor-critic."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("gymnasium")

from examples.train_tarmac import _collect_episode, _pad_episodes, _team_reward, train
from marl_envs import make_env
from modmarl.algorithms.tarmac import (
    TarMACAgent,
    TarMACConfig,
    TarMACCritic,
    TarMACPolicy,
)


def _small_config(**changes) -> TarMACConfig:
    return replace(
        TarMACConfig(hidden_dim=32, message_dim=16, signature_dim=8),
        **changes,
    )


def test_policy_shapes_and_categorical_actions() -> None:
    policy = TarMACPolicy(5, 4, _small_config())
    obs = torch.randn(6, 3, 5)
    state = policy.initial_state(6, 3, obs.device)
    logits, next_state, attention = policy(obs, state)
    assert logits.shape == (6, 3, 4)
    assert next_state.shape == (6, 3, 48)
    assert attention.shape == (6, 3, 3)
    action, log_prob, entropy, written, _ = policy.act(obs, state)
    assert action.shape == log_prob.shape == entropy.shape == (6, 3)
    assert written.shape == (6, 3, 48)
    assert action.dtype == torch.long


def test_attention_matches_scaled_signature_query_equation() -> None:
    torch.manual_seed(0)
    policy = TarMACPolicy(5, 4, _small_config())
    hidden = torch.randn(2, 3, 32)
    message, attention = policy._target(hidden)
    query = policy.query(hidden)
    signature = policy.signature(hidden)
    value = policy.value(hidden)
    expected_attention = torch.softmax(query @ signature.transpose(-1, -2) / (8 ** 0.5), dim=-1)
    torch.testing.assert_close(attention, expected_attention)
    torch.testing.assert_close(message, expected_attention @ value)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 3))


def test_single_round_message_is_consumed_at_next_timestep() -> None:
    torch.manual_seed(0)
    policy = TarMACPolicy(5, 4, _small_config(communication_rounds=1))
    obs = torch.randn(2, 3, 5)
    state = policy.initial_state(2, 3, obs.device)
    changed = obs.clone()
    changed[:, 2] += 1.0
    logits, written, _ = policy(obs, state)
    changed_logits, changed_written, _ = policy(changed, state)
    torch.testing.assert_close(logits[:, :2], changed_logits[:, :2])
    next_obs = torch.randn_like(obs)
    next_logits, _, _ = policy(next_obs, written)
    changed_next_logits, _, _ = policy(next_obs, changed_written)
    assert not torch.allclose(next_logits[:, 0], changed_next_logits[:, 0])
    assert not torch.allclose(next_logits[:, 1], changed_next_logits[:, 1])


def test_extra_round_updates_hidden_before_action() -> None:
    torch.manual_seed(0)
    policy = TarMACPolicy(5, 4, _small_config(communication_rounds=2))
    obs = torch.randn(2, 3, 5)
    state = policy.initial_state(2, 3, obs.device)
    changed = obs.clone()
    changed[:, 2] += 1.0
    logits, _, _ = policy(obs, state)
    changed_logits, _, _ = policy(changed, state)
    assert not torch.allclose(logits[:, 0], changed_logits[:, 0])
    assert not torch.allclose(logits[:, 1], changed_logits[:, 1])


def test_centralized_critic_consumes_hidden_states_and_all_actions() -> None:
    critic = TarMACCritic(3, 32, 4)
    hidden = torch.randn(6, 3, 32)
    actions = F.one_hot(torch.randint(0, 4, (6, 3)), 4).float()
    values = critic(hidden, actions)
    assert values.shape == (6,)
    changed_hidden = hidden.clone()
    changed_hidden[:, 2] += 1.0
    changed_actions = actions.roll(1, dims=-1)
    assert not torch.allclose(values, critic(changed_hidden, actions))
    assert not torch.allclose(values, critic(hidden, changed_actions))


def test_update_matches_paper_td_and_joint_policy_gradient() -> None:
    torch.manual_seed(0)
    config = _small_config(learning_rate=0.0, entropy_coefficient=0.01)
    agent = TarMACAgent(2, 3, 3, config)
    obs = torch.randn(2, 3, 2, 3)
    actions = torch.randint(0, 3, (2, 3, 2))
    rewards = torch.randn(2, 3)
    mask = torch.tensor([[1.0, 1, 1], [1, 1, 0]])
    continuation = torch.tensor([[1.0, 1, 0], [1, 0, 0]])

    state = agent.policy.initial_state(2, 2, obs.device)
    log_probs, entropies, hidden = [], [], []
    for step in range(3):
        logits, state, _ = agent.policy(obs[:, step], state)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs.append(distribution.log_prob(actions[:, step]))
        entropies.append(distribution.entropy())
        hidden.append(state[..., :32])
    hidden_tensor = torch.stack(hidden, dim=1).detach().reshape(6, 2, 32)
    one_hot = F.one_hot(actions, 3).float().reshape(6, 2, 3)
    q_values = agent.critic(hidden_tensor, one_hot).view(2, 3)
    next_q = torch.cat([q_values[:, 1:].detach(), torch.zeros_like(q_values[:, :1])], dim=1)
    target = rewards + 0.99 * continuation * next_q
    denominator = mask.sum()
    expected_critic = (((q_values - target).square() * mask).sum() / denominator).item()
    joint_log = torch.stack(log_probs, dim=1).sum(dim=-1)
    joint_entropy = torch.stack(entropies, dim=1).sum(dim=-1)
    expected_policy = -(
        (joint_log * q_values.detach() + 0.01 * joint_entropy) * mask
    ).sum().item() / denominator.item()

    update = agent.update(obs, actions, rewards, mask, continuation)
    assert update.critic_loss == pytest.approx(expected_critic)
    assert update.policy_loss == pytest.approx(expected_policy)


def test_message_value_receives_delayed_policy_gradient() -> None:
    torch.manual_seed(1)
    agent = TarMACAgent(3, 5, 4, _small_config(learning_rate=0.0))
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
        list(agent.critic.parameters())[-1].fill_(1.0)
    agent.update(
        torch.randn(2, 3, 3, 5),
        torch.randint(0, 4, (2, 3, 3)),
        torch.randn(2, 3),
        torch.ones(2, 3),
        torch.tensor([[1.0, 1, 0], [1, 1, 0]]),
    )
    gradient = agent.policy.value.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_optimizer_and_reported_defaults_match_paper() -> None:
    agent = TarMACAgent(3, 5, 4)
    group = agent.optimizer.param_groups[0]
    assert isinstance(agent.optimizer, torch.optim.RMSprop)
    assert group["lr"] == pytest.approx(7e-4)
    assert group["alpha"] == pytest.approx(0.99)
    assert agent.config.gamma == 0.99
    assert agent.config.entropy_coefficient == 0.01
    assert agent.config.hidden_dim == 128
    assert agent.config.message_dim == 32
    assert agent.config.signature_dim == 16
    assert agent.config.communication_rounds == 1
    assert not hasattr(agent, "target_policy")
    assert not hasattr(agent, "target_critic")


def test_global_reward_contract_and_finite_horizon_padding() -> None:
    assert _team_reward(2.0) == 2.0
    with pytest.raises(ValueError, match="global team reward"):
        _team_reward([1.0, 2.0])
    environment = make_env("navigation", 2, 4, 5)
    agent = TarMACAgent(2, environment.obs_dim, environment.num_actions, _small_config())
    episode, _ = _collect_episode(agent, environment, 2, 5, torch.device("cpu"))
    batch = _pad_episodes([episode], 2, environment.obs_dim, torch.device("cpu"))
    assert batch["obs"].shape == (1, 4, 2, environment.obs_dim)
    assert batch["rewards"].shape == (1, 4)
    assert batch["continuation"][0, -1] == 0


def test_two_round_signaling_rollout_uses_current_step_communication() -> None:
    environment = make_env("target_signaling", 3, 1, 5)
    agent = TarMACAgent(3, environment.obs_dim, environment.num_actions, _small_config(
        communication_rounds=2,
    ))
    episode, _ = _collect_episode(agent, environment, 3, 5, torch.device("cpu"))
    assert episode["obs"].shape == (1, 3, environment.obs_dim)
    assert agent.policy.config.communication_rounds == 2


def test_signaling_validation_requires_eighty_percent_success() -> None:
    summary = train(
        env="target_signaling",
        n_agents=3,
        horizon=1,
        episodes=2,
        seed=5,
        config=_small_config(communication_rounds=2),
        batch_episodes=2,
        evaluation_episodes=1,
        checkpoint=None,
    )
    assert summary["validation_criterion"] == {
        "final_success_rate": 0.8,
        "scope": "every confirmation seed",
    }


def test_policy_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    policy = TarMACPolicy(5, 4, _small_config(communication_rounds=2))
    obs = torch.randn(2, 3, 5)
    state = torch.randn(2, 3, 48)
    permutation = torch.tensor([2, 0, 1])
    logits, next_state, attention = policy(obs, state)
    permuted_logits, permuted_state, permuted_attention = policy(
        obs[:, permutation], state[:, permutation],
    )
    torch.testing.assert_close(permuted_logits, logits[:, permutation])
    torch.testing.assert_close(permuted_state, next_state[:, permutation])
    torch.testing.assert_close(
        permuted_attention,
        attention[:, permutation][:, :, permutation],
    )


def test_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "tarmac.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=3,
        seed=5,
        config=_small_config(),
        batch_episodes=2,
        evaluation_episodes=1,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "tarmac"
    assert summary["config"]["learner"] == "synchronous_centralized_actor_critic"
    assert summary["communication_rate"] == 1.0
    assert checkpoint.exists()
    checkpoint_state = torch.load(checkpoint, weights_only=True)
    assert "policy.signature.weight" in checkpoint_state["agent"]
