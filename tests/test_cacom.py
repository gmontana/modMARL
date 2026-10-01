"""Behavioral tests for CACOM's two stages, quantization, gating, and training."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples import train_cacom
from examples.train_cacom import train
from modmarl.algorithms.cacom import CACOMAgent, CACOMNetwork, LearnedStepQuantizer
from modmarl.common.replay import EpisodeReplayBuffer


def _agent(**overrides) -> CACOMAgent:
    kwargs = dict(
        n_agents=3, obs_dim=4, action_dim=5, hidden_dim=8, encode_dim=4,
        request_dim=2, response_dim=4, mixer_hidden_dim=6,
    )
    kwargs.update(overrides)
    return CACOMAgent(**kwargs)


def _batch():
    replay = EpisodeReplayBuffer(capacity=3, horizon=4, n_agents=3, obs_dim=4)
    replay.add_episode(
        obs=np.random.randn(5, 3, 4).astype(np.float32),
        actions=np.random.randint(0, 5, (4, 3)),
        rewards=np.random.randn(4).astype(np.float32),
        dones=np.zeros(4, dtype=np.float32),
    )
    return replay.sample(1, torch.device("cpu"))


def test_evaluation_records_success_for_the_signaling_task() -> None:
    from examples.train_cacom import train

    result = train(env="delayed_signaling", n_agents=2, horizon=3, episodes=0,
                   evaluation_episodes=16)
    for name in ("initial_evaluation", "random_evaluation", "final_evaluation"):
        evaluation = result[name]
        assert evaluation["successes"] == [float(value == 1.0) for value in evaluation["returns"]]
        assert len(evaluation["mean_distances"]) == 16


def test_lsq_outputs_representable_codes_and_has_step_gradient() -> None:
    quantizer = LearnedStepQuantizer(bits=2)
    values = torch.tensor([-4.0, -0.4, 0.4, 4.0], requires_grad=True)
    output = quantizer(values)
    assert set(output.detach().tolist()) <= {-1.0, 0.0, 1.0}
    output.sum().backward()
    assert values.grad is not None
    assert quantizer.step_size.grad is not None


def test_encode_treats_entities_as_tokens_and_quantizes_requests() -> None:
    network = CACOMNetwork(
        3, 6, 5, entity_schema=((1, 2), (1, 1), (1, 3)), encode_dim=4,
        request_dim=2, response_dim=4, hidden_dim=8,
    )
    features, requests = network.encode(
        torch.randn(2, 3, 6), network.init_hidden(2, torch.device("cpu")),
    )
    assert features.shape == (2, 3, 3, 4)
    assert requests.shape == (2, 3, 2)


def test_navigation_controller_inputs_have_paper_entity_partition() -> None:
    schema = train_cacom._entity_schema("navigation", n_agents=3, action_dim=5)
    assert schema == ((1, 2), (1, 2), (3, 2), (2, 2), (1, 5), (1, 3))
    assert sum(count * length for count, length in schema) == 14 + 5 + 3


def test_same_type_entities_share_one_encoder() -> None:
    # The release builds one Linear per entity TYPE (obs_segs) and applies it to every
    # token of that type; a per-token encoder would break permutation equivariance.
    network = CACOMNetwork(
        3, 10, 5, entity_schema=((1, 2), (4, 2)), encode_dim=4,
        request_dim=2, response_dim=4, hidden_dim=8,
    )
    assert len(network.entity_encoders) == 2
    assert network.n_entities == 5

    obs = torch.randn(2, 3, 10)
    features, _ = network.encode(obs, network.init_hidden(2, torch.device("cpu")))
    swapped = obs.clone()
    swapped[..., 2:4], swapped[..., 4:6] = obs[..., 4:6], obs[..., 2:4]
    swapped_features, _ = network.encode(
        swapped, network.init_hidden(2, torch.device("cpu")),
    )
    # Swapping two same-type tokens permutes their encodings rather than changing them.
    torch.testing.assert_close(features[:, :, 1], swapped_features[:, :, 2])
    torch.testing.assert_close(features[:, :, 2], swapped_features[:, :, 1])


def test_helper_response_is_personalized_by_receiver_request() -> None:
    torch.manual_seed(2)
    network = _agent().network
    hidden = network.init_hidden(1, torch.device("cpu"))
    features, requests = network.encode(torch.randn(1, 3, 4), hidden)
    changed = requests.clone()
    changed[:, 1] += 2.0
    first = network.personalized_responses(features, requests)
    second = network.personalized_responses(features, changed)
    assert not torch.allclose(first[:, 0, 1], second[:, 0, 1])
    assert torch.allclose(first[:, 0, 2], second[:, 0, 2])


def test_communication_has_helper_receiver_axes_and_removes_self_links() -> None:
    network = _agent().network
    hidden = network.init_hidden(2, torch.device("cpu"))
    features, requests = network.encode(torch.randn(2, 3, 4), hidden)
    received, logits, mask = network.communication(features, requests, force_all_links=True)
    assert logits.shape == mask.shape[:-1] + (2,)
    assert received.shape == (2, 3, 2, 4)
    diagonal = mask[:, torch.arange(3), torch.arange(3)]
    assert torch.count_nonzero(diagonal) == 0
    assert torch.all(mask.sum(dim=(1, 2, 3)) == 6)


def test_forced_link_mask_removes_exact_helper_to_receiver_message() -> None:
    network = _agent().network
    hidden = network.init_hidden(1, torch.device("cpu"))
    features, requests = network.encode(torch.randn(1, 3, 4), hidden)
    forced = torch.ones(1, 3, 3, 1)
    forced[:, 0, 2] = 0
    received, _, mask = network.communication(features, requests, forced_mask=forced)
    # Receiver 2 stores helpers [0, 1], so slot 0 is the removed 0 -> 2 link.
    assert torch.count_nonzero(received[:, 2, 0]) == 0
    assert mask[0, 0, 2].item() == 0


def test_step_shapes_and_helper_value_objective_reaches_message_path() -> None:
    agent = _agent()
    hidden = agent.init_hidden(2, torch.device("cpu"))
    q_values, next_hidden, products = agent.network.step(
        torch.randn(2, 3, 4), hidden, force_all_links=True,
    )
    loss = agent.network.helper_value_loss(products, q_values)
    loss.mean().backward()
    assert q_values.shape == (2, 3, 5)
    assert next_hidden.shape == (2, 3, 8)
    assert any(p.grad is not None for p in agent.network.predict_q.parameters())
    assert any(p.grad is not None for p in agent.network.response_head.parameters())


def test_gate_labels_train_gate_without_policy_gradients() -> None:
    agent = _agent()
    logits, labels = agent.network.gate_labels(
        torch.randn(2, 3, 4), agent.init_hidden(2, torch.device("cpu")), helper=1,
    )
    F = torch.nn.functional
    F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1)).backward()
    assert logits.shape == (2, 2, 2)
    assert labels.shape == (2, 2)
    assert any(parameter.grad is not None for parameter in agent.gate_parameters())
    assert all(parameter.grad is None for parameter in agent.policy_parameters())


def test_paper_gate_compares_actions_under_one_mixed_value_function(monkeypatch) -> None:
    agent = _agent()
    q_on = torch.tensor([[[0., 0., 0., 0., 0.], [2., 1., 0., 0., 0.], [2., 1., 0., 0., 0.]]])
    q_off = torch.tensor([[[0., 0., 0., 0., 0.], [1., 3., 0., 0., 0.], [3., 1., 0., 0., 0.]]])
    obs = torch.randn(1, 3, 4)
    hidden = agent.init_hidden(1, torch.device("cpu"))

    class WeightedMixer(torch.nn.Module):
        def forward(self, values, state):
            return (values * values.new_tensor([1., 2., 3.])).sum(-1, keepdim=True)

    def labels(mode, threshold):
        outputs = iter([q_on, q_off])
        monkeypatch.setattr(agent.network, "policy", lambda *args: (next(outputs), hidden))
        return agent.network.gate_labels(obs, hidden, 0, threshold, mode=mode,
                                         mixer=WeightedMixer(), state=obs.flatten(1))[1]

    # Release compares 2 - 3 and prunes even though the no-message action is worse.
    assert labels("release", 0).tolist() == [[1, 1]]
    # Paper changes only receiver 1's action: its mixed-value gain is 2*(2-1)=2.
    # Receiver 2 selects the same action, so its action-value gain is zero.
    assert labels("paper", 1.5).tolist() == [[0, 1]]


def test_policy_update_and_gate_update_are_finite_and_separate(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    policy_optimizer = torch.optim.Adam(agent.policy_parameters(), lr=1e-3)
    gate_optimizer = torch.optim.Adam(agent.gate_parameters(), lr=1e-3)
    helpers = []
    gate_labels = agent.network.gate_labels

    def record_helper(obs, hidden, helper, threshold=0.0):
        helpers.append(helper)
        return gate_labels(obs, hidden, helper, threshold)

    monkeypatch.setattr(agent.network, "gate_labels", record_helper)
    losses = train_cacom._update_policy(
        agent, policy_optimizer, _batch(), 0.95, 0.1, force_all_links=True,
    )
    gate_loss = train_cacom._update_gate(agent, gate_optimizer, _batch(), helper=0)
    assert all(np.isfinite(value) for value in losses.values())
    assert np.isfinite(gate_loss)
    assert helpers and set(helpers) == {0}


def test_gate_uses_the_released_rmsprop_optimizer() -> None:
    agent = _agent()
    optimizer = train_cacom._gate_optimizer(agent.gate_parameters(), 1e-4)
    assert isinstance(optimizer, torch.optim.RMSprop)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.defaults["alpha"] == 0.99
    assert optimizer.defaults["eps"] == 1e-5


def test_target_update_copies_network_and_mixer() -> None:
    agent = _agent()
    assert isinstance(agent.mixer.hyper_w1, torch.nn.Sequential)
    with torch.no_grad():
        next(agent.network.parameters()).add_(1.0)
        next(agent.mixer.parameters()).add_(1.0)
    agent.update_targets()
    assert all(torch.equal(a, b) for a, b in zip(agent.network.parameters(), agent.target_network.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(agent.mixer.parameters(), agent.target_mixer.parameters()))


def test_cacom_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "cacom.pt"
    summary = train(
        env="navigation", n_agents=3, horizon=5, episodes=5, seed=4,
        hidden_dim=8, encode_dim=4, request_dim=2, response_dim=4,
        buffer_episodes=8, batch_episodes=2, warmup_episodes=2,
        gate_start_steps=10, gate_update_every_steps=1, checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "cacom"
    assert summary["gate_updates"] > 0
    assert len(summary["evaluation_returns"]) == 20
    assert checkpoint.exists()


def test_quantizer_matches_the_released_lsq_formulation() -> None:
    # Released `LsqQuantizer`: symmetric bounds, grad_scale = 1/sqrt(thd_pos * numel),
    # and the divide -> clamp -> round(straight-through) -> multiply order. The paper's
    # Eq. (4) instead writes an asymmetric range; the release governs the channel.
    torch.manual_seed(19)
    quantizer = LearnedStepQuantizer(bits=3).double()
    with torch.no_grad():
        quantizer.step_size.copy_(torch.tensor([0.37], dtype=torch.float64))
    value = torch.randn(4, 3, 6, dtype=torch.float64) * 2.0

    thd_pos = 2 ** (3 - 1) - 1
    thd_neg = -(2 ** (3 - 1)) + 1
    grad_scale = 1.0 / math.sqrt(thd_pos * value.numel())
    step = quantizer.step_size.abs().clamp_min(1e-8)
    step = (step - step * grad_scale).detach() + step * grad_scale
    scaled = (value / step).clamp(thd_neg, thd_pos)
    expected = ((scaled.round() - scaled).detach() + scaled) * step

    torch.testing.assert_close(quantizer(value), expected, rtol=0.0, atol=1e-14)


def test_target_network_keeps_its_gate_at_initialization() -> None:
    # The release owns ExpGate on the controller, outside the agent, and `load_state`
    # copies the agent only — so the target gate never advances past initialization.
    agent = _agent()
    gate_before = [p.detach().clone() for p in agent.target_network.gate_head.parameters()]
    with torch.no_grad():
        for parameter in agent.gate_parameters():
            parameter.add_(1.0)
        next(agent.network.entity_kqv.parameters()).add_(1.0)
    agent.update_targets()

    for before, after in zip(gate_before, agent.target_network.gate_head.parameters()):
        assert torch.equal(before, after)
    torch.testing.assert_close(
        next(agent.network.entity_kqv.parameters()),
        next(agent.target_network.entity_kqv.parameters()),
    )
