"""Golden tests for MAGIC's released graph equations and rollout learner."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_magic import _agent_rewards, _collect_episode, _pad_episodes, train
from marl_envs import make_env
from modmarl.algorithms.magic import (
    GATMessageProcessor,
    GraphAttentionScheduler,
    MAGICAgent,
    MAGICConfig,
    SelfLoopMode,
)


def _small_config(**changes) -> MAGICConfig:
    config = MAGICConfig(
        hidden_dim=16,
        gat_hidden_dim=4,
        gat_heads=2,
        use_gat_encoder=True,
        gat_encoder_dim=8,
        gat_encoder_heads=2,
        detach_gap=2,
    )
    return replace(config, **changes)


def test_default_configuration_is_released_predator_prey_medium() -> None:
    config = MAGICConfig()
    assert config.hidden_dim == 128
    assert config.gat_hidden_dim == 32
    assert config.gat_heads == 4
    assert config.directed
    assert config.use_gat_encoder
    assert config.gat_encoder_dim == 32
    assert config.gat_encoder_heads == 8
    assert config.learn_second_graph
    assert config.first_normalize and config.second_normalize
    assert config.first_self_loop is SelfLoopMode.LEARNED
    assert config.value_coefficient == pytest.approx(0.015)


def test_scheduler_adjacency_is_binary_differentiable_and_alive_masked() -> None:
    torch.manual_seed(0)
    scheduler = GraphAttentionScheduler(16)
    features = torch.randn(4, 3, 16, requires_grad=True)
    alive = torch.tensor([[1.0, 1.0, 0.0]]).expand(4, -1)
    adjacency, noise = scheduler(features, alive)
    assert adjacency.shape == (4, 3, 3)
    assert noise.shape == (4, 3, 3, 2)
    assert torch.all((adjacency.detach() == 0.0) | (adjacency.detach() == 1.0))
    assert torch.count_nonzero(adjacency[:, 2]) == 0
    assert torch.count_nonzero(adjacency[:, :, 2]) == 0
    adjacency.sum().backward()
    assert features.grad is not None


def test_scheduler_replays_gumbel_noise_exactly() -> None:
    torch.manual_seed(0)
    scheduler = GraphAttentionScheduler(16)
    features = torch.randn(4, 3, 16)
    adjacency, noise = scheduler(features)
    replayed, _ = scheduler(features, noise=noise)
    torch.testing.assert_close(replayed, adjacency)


def test_undirected_scheduler_symmetrizes_logits_before_sampling() -> None:
    torch.manual_seed(0)
    scheduler = GraphAttentionScheduler(8, directed=False)
    features = torch.randn(2, 3, 8)
    noise = torch.randn(2, 3, 3, 2)
    adjacency, _ = scheduler(features, noise=noise)
    receiver = features.unsqueeze(2).expand(-1, -1, 3, -1)
    sender = features.unsqueeze(1).expand(-1, 3, -1, -1)
    pair = torch.cat([receiver, sender], dim=-1)
    logits = 0.5 * scheduler.mlp(pair) + 0.5 * scheduler.mlp(pair.transpose(1, 2))
    expected = ((logits + noise).argmax(dim=-1) == 1).to(adjacency.dtype)
    torch.testing.assert_close(adjacency.detach(), expected)


@pytest.mark.parametrize("normalize", [False, True])
def test_gat_matches_released_attention_equation(normalize: bool) -> None:
    torch.manual_seed(0)
    gat = GATMessageProcessor(
        6, 5, num_heads=2, normalize=normalize, self_loop=SelfLoopMode.LEARNED,
    ).double()
    messages = torch.randn(1, 4, 6, dtype=torch.float64)
    adjacency = torch.tensor(
        [[0.0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0]],
        dtype=torch.float64,
    ).unsqueeze(0)
    output = gat(messages, adjacency)

    for head in range(2):
        transformed = messages[0] @ gat.weight[head]
        scores = torch.nn.functional.leaky_relu(
            (transformed @ gat.attn_receiver[head]).unsqueeze(-1)
            + (transformed @ gat.attn_sender[head]).unsqueeze(0),
            0.2,
        )
        attention = torch.softmax(scores * adjacency[0], dim=-1) * adjacency[0]
        if normalize:
            attention = attention + 1e-15
            attention = attention / attention.sum(dim=-1, keepdim=True)
            attention = attention * adjacency[0]
        expected = attention @ transformed
        start = head * 5
        actual = output[0, :, start:start + 5] - gat.bias[start:start + 5]
        torch.testing.assert_close(actual, expected)


def test_gat_projection_uses_released_flattened_xavier_fans(monkeypatch) -> None:
    shapes = []
    original = torch.nn.init.xavier_normal_

    def record_shape(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.nn.init, "xavier_normal_", record_shape)
    GATMessageProcessor(6, 5, num_heads=2)
    assert shapes == [(6, 10), (2, 5, 1), (2, 5, 1)]


def test_gat_self_loop_controls_match_release() -> None:
    messages = torch.randn(1, 3, 4)
    adjacency = torch.zeros(1, 3, 3)
    with_loop = GATMessageProcessor(4, 4, self_loop=SelfLoopMode.WITH)
    without_loop = GATMessageProcessor(4, 4, self_loop=SelfLoopMode.WITHOUT)
    assert not torch.allclose(with_loop(messages, adjacency), with_loop.bias)
    torch.testing.assert_close(
        without_loop(messages, adjacency),
        without_loop.bias.expand(1, 3, 4),
    )


def test_second_round_reuses_first_graph_when_release_flag_is_disabled() -> None:
    agent = MAGICAgent(5, 4, _small_config(learn_second_graph=False))
    obs = torch.randn(2, 3, 5)
    hidden, cell = agent.init_state(2, 3, obs.device)
    *_, adjacency, noise = agent(obs, hidden, cell)
    torch.testing.assert_close(adjacency[1], adjacency[0])
    torch.testing.assert_close(noise[1], noise[0])
    assert agent.second_scheduler is None


def test_agent_forward_uses_released_gat_encoder_and_shapes() -> None:
    agent = MAGICAgent(5, 4, _small_config())
    obs = torch.randn(6, 3, 5)
    hidden, cell = agent.init_state(6, 3, obs.device)
    logits, value, new_hidden, new_cell, adjacency, noise = agent(obs, hidden, cell)
    assert agent.gat_encoder is not None
    assert logits.shape == (6, 3, 4)
    assert value.shape == (6, 3)
    assert new_hidden.shape == new_cell.shape == (6, 3, 16)
    assert adjacency.shape == (2, 6, 3, 3)
    assert noise.shape == (2, 6, 3, 3, 2)
    assert agent.first_scheduler is not None
    assert agent.first_scheduler.mlp[2].in_features == 4
    assert agent.first_scheduler.mlp[2].out_features == 4


def test_recurrent_replay_reproduces_collected_log_probabilities() -> None:
    torch.manual_seed(0)
    agent = MAGICAgent(5, 4, _small_config())
    observations = torch.randn(4, 1, 3, 5)
    hidden, cell = agent.init_state(1, 3, observations.device)
    stored = []
    with torch.no_grad():
        for step in range(4):
            action, log_prob, _, hidden, cell, _, noise = agent.act(
                observations[step], hidden, cell,
            )
            stored.append((action, log_prob, noise))
    hidden, cell = agent.init_state(1, 3, observations.device)
    with torch.no_grad():
        for step, (action, expected, noise) in enumerate(stored):
            actual, _, _, hidden, cell = agent.evaluate_step(
                observations[step], hidden, cell, action, noise,
            )
            torch.testing.assert_close(actual, expected)


def test_individual_rewards_produce_individual_undiscounted_returns() -> None:
    torch.manual_seed(0)
    agent = MAGICAgent(2, 2, _small_config(learning_rate=0.0, use_gat_encoder=False))
    observations = torch.randn(1, 2, 2, 2)
    actions = torch.zeros(1, 2, 2, dtype=torch.long)
    rewards = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]])
    mask = torch.ones_like(rewards)
    noise = torch.zeros(1, 2, 2, 2, 2, 2)
    update = agent.update(observations, actions, rewards, mask, noise)
    assert update.return_mean == pytest.approx((3.0 + 30.0 + 2.0 + 20.0) / 4)


def test_policy_loss_reaches_both_released_schedulers() -> None:
    torch.manual_seed(1)
    agent = MAGICAgent(3, 3, _small_config(learning_rate=0.0, use_gat_encoder=False))
    batch, horizon, n_agents = 2, 3, 3
    observations = torch.randn(batch, horizon, n_agents, 3)
    actions = torch.randint(0, 3, (batch, horizon, n_agents))
    rewards = torch.randn(batch, horizon, n_agents)
    mask = torch.ones_like(rewards)
    noise = torch.randn(batch, horizon, 2, n_agents, n_agents, 2)
    agent.update(observations, actions, rewards, mask, noise)
    for scheduler in (agent.first_scheduler, agent.second_scheduler):
        assert scheduler is not None
        gradients = [parameter.grad for parameter in scheduler.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_optimizer_and_objective_defaults_match_release() -> None:
    agent = MAGICAgent(3, 3)
    group = agent.optimizer.param_groups[0]
    assert isinstance(agent.optimizer, torch.optim.RMSprop)
    assert group["lr"] == pytest.approx(1e-3)
    assert group["alpha"] == pytest.approx(0.97)
    assert group["eps"] == pytest.approx(1e-6)
    assert agent.config.gamma == 1.0
    assert agent.config.mean_ratio == 0.0
    assert agent.config.value_coefficient == 0.015
    assert agent.config.entropy_coefficient == 0.0
    assert agent.config.detach_gap == 10


def test_reward_adapter_and_episode_batch_keep_agent_axis() -> None:
    assert _agent_rewards(2.0, {}, 3).tolist() == [2.0, 2.0, 2.0]
    assert _agent_rewards(0.0, {"agent_rewards": [1.0, 2.0, 3.0]}, 3).tolist() == [1.0, 2.0, 3.0]
    environment = make_env("navigation", 2, 4, 5)
    agent = MAGICAgent(environment.obs_dim, environment.num_actions, _small_config())
    episode, _ = _collect_episode(agent, environment, 2, 5, torch.device("cpu"))
    batch = _pad_episodes([episode], 2, environment.obs_dim, torch.device("cpu"))
    assert batch["rewards"].shape == (1, 4, 2)
    assert batch["noise"].shape == (1, 4, 2, 2, 2, 2)


def test_permutation_equivariance_with_replayed_graph_noise() -> None:
    torch.manual_seed(0)
    agent = MAGICAgent(5, 4, _small_config())
    obs = torch.randn(2, 3, 5)
    hidden = torch.randn(2, 3, 16)
    cell = torch.randn(2, 3, 16)
    permutation = torch.tensor([2, 0, 1])
    *_, noise = agent(obs, hidden, cell)
    logits, value, *_ = agent(obs, hidden, cell, noise=noise)
    permuted_noise = noise[:, :, permutation][:, :, :, permutation]
    permuted_logits, permuted_value, *_ = agent(
        obs[:, permutation], hidden[:, permutation], cell[:, permutation], noise=permuted_noise,
    )
    torch.testing.assert_close(permuted_logits, logits[:, permutation], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(permuted_value, value[:, permutation], atol=1e-5, rtol=1e-5)


def test_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "magic.pt"
    summary = train(
        env="navigation",
        n_agents=2,
        horizon=4,
        episodes=3,
        seed=5,
        architecture=_small_config(),
        batch_steps=4,
        evaluation_episodes=1,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "magic"
    assert summary["episodes"] == 3
    assert summary["config"]["optimizer"] == "RMSprop"
    assert checkpoint.exists()
    checkpoint_state = torch.load(checkpoint, weights_only=True)
    assert "first_scheduler.mlp.0.weight" in checkpoint_state["agent"]
