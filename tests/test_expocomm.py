"""Behavioral tests for ExpoComm's topology, memory, auxiliary loss, and trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples import train_expocomm
from examples.train_expocomm import train
from modmarl.algorithms.expocomm import (
    ExpoCommAgent,
    ExpoCommNetwork,
    expo_contrastive_loss,
    exponential_offsets,
    exponential_peer_indices,
    static_exponential_peer_indices,
)
from modmarl.common.replay import EpisodeReplayBuffer


def _agent(**overrides) -> ExpoCommAgent:
    kwargs = dict(n_agents=5, obs_dim=4, action_dim=3, hidden_dim=8, mixer_hidden_dim=6)
    kwargs.update(overrides)
    return ExpoCommAgent(**kwargs)


def _batch(n_agents: int = 5, obs_dim: int = 4, horizon: int = 3):
    replay = EpisodeReplayBuffer(capacity=3, horizon=horizon, n_agents=n_agents, obs_dim=obs_dim)
    replay.add_episode(
        obs=np.random.randn(horizon + 1, n_agents, obs_dim).astype(np.float32),
        actions=np.random.randint(0, 3, (horizon, n_agents)),
        rewards=np.random.randn(horizon).astype(np.float32),
        dones=np.zeros(horizon, dtype=np.float32),
    )
    return replay.sample(1, torch.device("cpu"))


def test_exponential_offsets_cover_graph_in_logarithmic_rounds() -> None:
    assert exponential_offsets(1) == (0,)
    assert exponential_offsets(5) == (0, 1, 2, 4)
    assert len(exponential_offsets(100)) == 1 + 7


def test_exponential_peers_cycle_and_wrap() -> None:
    # Paper Equation (2) keeps the local stream persistent while rotating through
    # non-self power-of-two peers. The release's extra self-only round is omitted.
    expected = [
        torch.tensor([1, 2, 3, 4, 0]),
        torch.tensor([2, 3, 4, 0, 1]),
        torch.tensor([4, 0, 1, 2, 3]),
    ]
    for timestep, peers in enumerate(expected):
        assert torch.equal(exponential_peer_indices(5, timestep), peers)
    assert torch.equal(exponential_peer_indices(5, 3), expected[0])


def test_one_agent_one_peer_topology_falls_back_to_self() -> None:
    assert torch.equal(exponential_peer_indices(1, 9), torch.tensor([0]))


def test_static_topology_contains_self_and_every_power_of_two_peer() -> None:
    assert torch.equal(
        static_exponential_peer_indices(5),
        torch.tensor([
            [0, 1, 2, 4], [1, 2, 3, 0], [2, 3, 4, 1],
            [3, 4, 0, 2], [4, 0, 1, 3],
        ]),
    )


def test_static_network_attention_aggregates_exact_exponential_neighbors() -> None:
    network = ExpoCommNetwork(
        input_dim=4, action_dim=3, state_dim=20, hidden_dim=4, topology="static",
    )
    with torch.no_grad():
        network.message_query.weight.zero_()
        network.message_query.bias.zero_()
        network.message_key.weight.zero_()
        network.message_key.bias.zero_()
        network.message_value.weight.copy_(torch.eye(4))
        network.message_value.bias.zero_()
    messages = torch.arange(5.0).view(1, 5, 1).expand(-1, -1, 4)
    aggregated = network._static_messages(
        torch.zeros(1, 5, 4), messages, static_exponential_peer_indices(5),
    )
    assert aggregated[0, 0, 0].item() == pytest.approx((0 + 1 + 2 + 4) / 4)
    assert aggregated[0, 4, 0].item() == pytest.approx((4 + 0 + 1 + 3) / 4)


def test_static_messages_include_current_local_information() -> None:
    torch.manual_seed(0)
    network = ExpoCommNetwork(
        input_dim=4, action_dim=3, state_dim=15, hidden_dim=8,
        topology="static", attention_dim=4,
    )
    hidden = torch.zeros(1, 3, 8)
    messages = torch.zeros_like(hidden)
    peers = static_exponential_peer_indices(3)
    first = network.step(torch.zeros(1, 3, 4), hidden, messages, peers)[2]
    second = network.step(torch.randn(1, 3, 4), hidden, messages, peers)[2]
    assert not torch.allclose(first, second)
    assert not torch.allclose(second[:, 0], second[:, 1])


def test_network_reads_exact_selected_peer() -> None:
    torch.manual_seed(0)
    network = ExpoCommNetwork(input_dim=4, action_dim=3, state_dim=20, hidden_dim=8)
    obs = torch.zeros(1, 5, 4)
    hidden = torch.zeros(1, 5, 8)
    messages = torch.arange(5.0).view(1, 5, 1).expand(-1, -1, 8)
    peers = exponential_peer_indices(5, 2)

    captured = {}
    hook = network.message_input[0].register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("input", inputs[0].detach().clone()),
    )
    network.step(obs, hidden, messages, peers)
    hook.remove()
    received = captured["input"][..., 8:]
    assert torch.equal(received[:, :, 0], torch.tensor([[4.0, 0.0, 1.0, 2.0, 3.0]]))


def test_step_shapes_and_message_memory_changes() -> None:
    agent = _agent()
    hidden, messages = agent.init_recurrent(2, torch.device("cpu"))
    q_values, next_hidden, next_messages = agent.step(torch.randn(2, 5, 4), hidden, messages, 1)
    assert q_values.shape == (2, 5, 3)
    assert next_hidden.shape == next_messages.shape == (2, 5, 8)
    assert not torch.equal(messages, next_messages)


def test_auxiliary_loss_reaches_message_processor_and_predictor() -> None:
    agent = _agent()
    hidden, messages = agent.init_recurrent(2, torch.device("cpu"))
    _, _, messages = agent.step(torch.randn(2, 5, 4), hidden, messages, 1)
    agent.predict_state(messages).pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.network.message_gru.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.network.state_predictor.parameters())
    assert all(p.grad is None for p in agent.network.q_head.parameters())


def test_contrastive_grounding_rewards_same_timestep_agreement() -> None:
    messages = torch.eye(6).view(1, 6, 1, 6).expand(-1, -1, 3, -1).clone()
    aligned = expo_contrastive_loss(messages, mask=None, diameter=1)
    permuted = messages.clone()
    permuted[:, :, 1] = messages[:, torch.tensor([3, 4, 5, 0, 1, 2]), 1]
    misaligned = expo_contrastive_loss(permuted, mask=None, diameter=1)
    assert aligned < misaligned


def test_contrastive_negative_sampling_is_seeded() -> None:
    messages = torch.randn(2, 12, 3, 8)
    first = expo_contrastive_loss(
        messages, mask=None, diameter=1,
        generator=torch.Generator().manual_seed(7),
    )
    second = expo_contrastive_loss(
        messages, mask=None, diameter=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(first, second)


def test_agent_selects_static_topology_and_contrastive_grounding() -> None:
    agent = _agent(topology="static", grounding="contrastive")
    hidden, messages = agent.init_recurrent(1, torch.device("cpu"))
    q_values, _, _ = agent.step(torch.randn(1, 5, 4), hidden, messages, 0)
    assert q_values.shape == (1, 5, 3)
    assert isinstance(agent.mixer.hyper_w1, torch.nn.Sequential)
    sequence = torch.randn(2, 8, 5, 8)
    assert torch.isfinite(agent.grounding_loss(sequence, mask=torch.ones(2, 8)))


def test_update_is_finite_and_does_not_change_targets() -> None:
    torch.manual_seed(0)
    agent = _agent()
    optimizer = torch.optim.Adam(list(agent.network.parameters()) + list(agent.mixer.parameters()), lr=1e-3)
    targets = [parameter.clone() for parameter in agent.target_network.parameters()]
    losses = train_expocomm._update(agent, optimizer, _batch(), gamma=0.95, aux_coef=0.1)
    assert all(np.isfinite(value) for value in losses.values())
    assert losses["loss"] == pytest.approx(
        losses["td_loss"] + 0.1 * losses["aux_loss"],
    )
    for before, after in zip(targets, agent.target_network.parameters()):
        assert torch.equal(before, after)


def test_static_contrastive_update_is_finite() -> None:
    torch.manual_seed(0)
    agent = _agent(topology="static", grounding="contrastive")
    optimizer = torch.optim.Adam(list(agent.network.parameters()) + list(agent.mixer.parameters()), lr=1e-3)
    losses = train_expocomm._update(agent, optimizer, _batch(horizon=6), gamma=0.95, aux_coef=0.1)
    assert all(np.isfinite(value) for value in losses.values())
    assert losses["loss"] == pytest.approx(1.1 * losses["td_loss"])


def test_padding_does_not_change_auxiliary_loss(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent(n_agents=2, obs_dim=3)
    replay = EpisodeReplayBuffer(capacity=2, horizon=3, n_agents=2, obs_dim=3)
    replay.add_episode(
        obs=np.random.randn(2, 2, 3).astype(np.float32),
        actions=np.random.randint(0, 3, (1, 2)),
        rewards=np.zeros(1, dtype=np.float32),
        dones=np.ones(1, dtype=np.float32),
    )
    batch = replay.sample(1, torch.device("cpu"))
    optimizer = torch.optim.Adam(list(agent.network.parameters()) + list(agent.mixer.parameters()), lr=0.0)
    first = train_expocomm._update(agent, optimizer, batch, gamma=0.95, aux_coef=0.1)["aux_loss"]
    batch.obs[:, 2:] = 10_000.0
    second = train_expocomm._update(agent, optimizer, batch, gamma=0.95, aux_coef=0.1)["aux_loss"]
    assert second == pytest.approx(first)


def test_target_update_copies_network_and_mixer() -> None:
    agent = _agent()
    with torch.no_grad():
        next(agent.network.parameters()).add_(1.0)
        next(agent.mixer.parameters()).add_(1.0)
    agent.update_targets()
    for online, target in zip(agent.network.parameters(), agent.target_network.parameters()):
        assert torch.equal(online, target)
    for online, target in zip(agent.mixer.parameters(), agent.target_mixer.parameters()):
        assert torch.equal(online, target)


def test_expocomm_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "expocomm.pt"
    summary = train(
        env="navigation", n_agents=3, horizon=5, episodes=4, seed=5,
        hidden_dim=12, mixer_hidden_dim=8, buffer_episodes=8,
        batch_episodes=2, warmup_episodes=2, checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "expocomm"
    assert len(summary["evaluation_returns"]) == 20
    assert summary["communication_rate"] == 0.5
    assert checkpoint.exists()
    state = torch.load(checkpoint, weights_only=True)
    for prefix in ("network", "mixer", "target_network", "target_mixer"):
        assert any(name.startswith(prefix) for name in state)


def test_running_reward_stats_accumulate_across_batches() -> None:
    stats = train_expocomm._RunningRewardStats(torch.device("cpu"))
    stats.update(torch.tensor([[1.0, 3.0]]), torch.ones(1, 2))
    first_mean = stats.mean.clone()
    stats.update(torch.tensor([[5.0, 7.0]]), torch.ones(1, 2))
    assert torch.allclose(first_mean, torch.tensor(2.0), atol=2e-4)
    assert torch.allclose(stats.mean, torch.tensor(4.0), atol=2e-4)
    assert stats.var > 0


def test_state_grounding_loss_matches_the_released_normalizer() -> None:
    # Upstream `q_learner.py` sums the squared error over state_dim but normalizes by a
    # mask spanning only (batch, time, agent):
    #     aux_loss = (masked_predict_states_error ** 2).sum() / predict_mask.sum()
    # Expanding the mask over state_dim as well would shrink the loss by exactly
    # state_dim, silently weakening the grounding gradient.
    torch.manual_seed(37)
    n_agents, obs_dim, batch, steps = 4, 3, 2, 5
    agent = ExpoCommAgent(n_agents, obs_dim, 5, hidden_dim=8, grounding="state").double()

    messages = torch.randn(batch, steps, n_agents, 8, dtype=torch.float64)
    states = torch.randn(batch, steps, n_agents * obs_dim, dtype=torch.float64)
    mask = torch.ones(batch, steps, dtype=torch.float64)
    mask[0, -2:] = 0.0                       # padded tail on one episode

    got = agent.grounding_loss(messages, states=states, mask=mask)

    predictions = agent.network.predict_state(messages)
    error = (predictions - states.unsqueeze(2).expand_as(predictions)) ** 2
    released_mask = mask[:, :, None, None].expand(batch, steps, n_agents, 1)
    expected = (error * released_mask).sum() / released_mask.sum()

    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-12)
