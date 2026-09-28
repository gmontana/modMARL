"""Paper and PyMARL-alpha equation tests for QMIX."""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_qmix import train
from modmarl.algorithms.qmix import AgentQNetwork, QMIXAgent, QMIXBatch, QMixer, QMIXReplayBuffer


def _batch(batch_size=2, time_steps=3, n_agents=2, action_dim=3, state_dim=8):
    return QMIXBatch(
        obs=torch.randn(batch_size, time_steps + 1, n_agents, 4),
        states=torch.randn(batch_size, time_steps + 1, state_dim),
        actions=torch.randint(action_dim, (batch_size, time_steps, n_agents)),
        available_actions=torch.ones(batch_size, time_steps + 1, n_agents, action_dim),
        rewards=torch.randn(batch_size, time_steps), dones=torch.zeros(batch_size, time_steps),
        mask=torch.ones(batch_size, time_steps),
    )


def test_agent_network_is_recurrent_gru_utility() -> None:
    network = AgentQNetwork(9, 3, hidden_dim=16)
    q, hidden = network(torch.randn(4, 9), torch.zeros(4, 16))
    assert q.shape == (4, 3) and hidden.shape == (4, 16)
    assert isinstance(network.gru, torch.nn.GRUCell)


def test_mixer_uses_paper_single_layer_positive_hypernetworks() -> None:
    mixer = QMixer(3, 8, 16)
    assert isinstance(mixer.hyper_w1, torch.nn.Linear)
    assert isinstance(mixer.hyper_w2, torch.nn.Linear)
    utilities = torch.randn(5, 3, requires_grad=True)
    mixer(utilities, torch.randn(5, 8)).sum().backward()
    assert (utilities.grad >= -1e-7).all()


def test_two_layer_hypernetwork_matches_maic_qmix_extension() -> None:
    mixer = QMixer(3, 5, 8, hypernet_hidden_dim=64)
    assert isinstance(mixer.hyper_w1, torch.nn.Sequential)
    assert isinstance(mixer.hyper_w2, torch.nn.Sequential)
    utilities = torch.randn(2, 4, 3, requires_grad=True)
    values = mixer(utilities, torch.randn(2, 4, 5))
    assert values.shape == (2, 4)
    values.sum().backward()
    assert (utilities.grad >= -1e-7).all()


def test_joint_argmax_factorises_under_monotonic_mixer() -> None:
    mixer = QMixer(2, 5, 8)
    utilities = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    state = torch.randn(1, 5)
    values = {}
    for first in range(2):
        for second in range(2):
            selected = torch.tensor([[utilities[0, first], utilities[1, second]]])
            values[(first, second)] = float(mixer(selected, state).detach())
    assert max(values, key=values.get) == (1, 0)


def test_double_q_uses_online_argmax_and_target_evaluation() -> None:
    learner = QMIXAgent(2, 4, 2, state_dim=6, hidden_dim=8, learning_rate=0.0, gamma=0.5)
    batch = _batch(batch_size=1, time_steps=1, n_agents=2, action_dim=2, state_dim=6)
    batch.rewards.fill_(1.0)
    online = torch.tensor(
        [[[[0.0, 0.0], [0.0, 0.0]], [[9.0, 1.0], [8.0, 2.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[3.0, 20.0], [4.0, 30.0]]]])

    def unroll(self, network, obs, actions):
        return online if network is self.q_network else target

    learner._unroll = types.MethodType(unroll, learner)
    learner.mixer.forward = lambda values, states: values.sum(-1)
    learner.target_mixer.forward = lambda values, states: values.sum(-1)
    metrics = learner.update(batch)
    torch.testing.assert_close(metrics["targets"], torch.tensor([[1.0 + 0.5 * 7.0]]))


def test_release_optimizer_and_hard_target_configuration() -> None:
    learner = QMIXAgent(2, 4, 3, state_dim=8, hidden_dim=8)
    assert isinstance(learner.optimizer, torch.optim.RMSprop)
    assert learner.optimizer.defaults["alpha"] == pytest.approx(0.99)
    assert learner.optimizer.defaults["eps"] == pytest.approx(1e-5)
    before = copy.deepcopy(learner.target_q_network.state_dict())
    with torch.no_grad():
        next(learner.q_network.parameters()).add_(1.0)
    assert learner.maybe_update_targets(199) is False
    assert all(torch.equal(value, before[key]) for key, value in learner.target_q_network.state_dict().items())
    assert learner.maybe_update_targets(200) is True


def test_previous_action_and_identity_are_part_of_unroll_input() -> None:
    learner = QMIXAgent(2, 4, 3, state_dim=8, hidden_dim=8)
    batch = _batch(batch_size=1, time_steps=2, n_agents=2, action_dim=3, state_dim=8)
    changed = copy.deepcopy(batch)
    changed.actions[:, 0] = (batch.actions[:, 0] + 1) % 3
    first = learner._unroll(learner.q_network, batch.obs, batch.actions)
    second = learner._unroll(learner.q_network, changed.obs, changed.actions)
    torch.testing.assert_close(first[:, 0], second[:, 0])
    assert not torch.allclose(first[:, 1], second[:, 1])


def test_update_changes_online_but_not_target_parameters() -> None:
    learner = QMIXAgent(2, 4, 3, state_dim=8, hidden_dim=8, learning_rate=1e-2)
    online_before = copy.deepcopy(learner.q_network.state_dict())
    target_before = copy.deepcopy(learner.target_q_network.state_dict())
    learner.update(_batch())
    assert any(not torch.equal(value, online_before[key]) for key, value in learner.q_network.state_dict().items())
    assert all(torch.equal(value, target_before[key]) for key, value in learner.target_q_network.state_dict().items())


def test_qmix_episode_replay_retains_state_availability_and_padding() -> None:
    replay = QMIXReplayBuffer(4, 5, 2, 3, 7, 4)
    replay.add_episode(
        obs=np.ones((4, 2, 3), np.float32), states=np.ones((4, 7), np.float32),
        actions=np.ones((3, 2), np.int64), available_actions=np.ones((4, 2, 4), np.float32),
        rewards=np.ones(3, np.float32), dones=np.array([0, 0, 1], np.float32),
    )
    batch = replay.sample(1, torch.device("cpu"))
    assert batch.states.shape == (1, 6, 7)
    assert batch.available_actions.shape == (1, 6, 2, 4)
    torch.testing.assert_close(batch.mask[0], torch.tensor([1., 1., 1., 0., 0.]))


def test_qmix_train_smoke_and_complete_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "qmix.pt"
    summary = train(
        env="navigation", n_agents=3, horizon=6, episodes=5, seed=5,
        hidden_dim=8, mixer_hidden_dim=8, buffer_size=16, batch_size=2,
        warmup_episodes=2, target_update_interval=2, evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "qmix"
    saved = torch.load(checkpoint, weights_only=True)
    assert "model" in saved and "optimizer" in saved
    assert "q_network.gru.weight_ih" in saved["model"]
    assert "mixer.hyper_w1.weight" in saved["model"]
