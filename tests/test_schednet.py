from __future__ import annotations

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_schednet import train
from modmarl.algorithms.schednet import (
    ActionSelector,
    MessageEncoder,
    SchedNetAgent,
    SchedNetCritic,
    WeightGenerator,
    aggregate_broadcast,
    top_k_schedule,
)


def test_weight_generator_shapes() -> None:
    generator = WeightGenerator(n_agents=3, obs_dim=5, hidden_dim=32)
    weights = generator(torch.randn(6, 3, 5))
    assert tuple(weights.shape) == (6, 3)
    assert torch.all(weights >= 0.0) and torch.all(weights <= 1.0)


def test_message_encoder_shapes() -> None:
    encoder = MessageEncoder(n_agents=3, obs_dim=5, message_dim=2, hidden_dim=32)
    messages = encoder(torch.randn(6, 3, 5))
    assert tuple(messages.shape) == (6, 3, 2)


def test_action_selector_shapes() -> None:
    selector = ActionSelector(
        n_agents=3, obs_dim=5, channel_dim=2, action_dim=4, hidden_dim=32,
    )
    obs = torch.randn(6, 3, 5)
    message = torch.zeros(6, 3, 2)
    logits = selector(obs, message)
    assert tuple(logits.shape) == (6, 3, 4)
    action, log_prob = selector.sample(obs, message)
    assert tuple(action.shape) == (6, 3)
    assert tuple(log_prob.shape) == (6, 3)
    assert torch.all(action < 4)


def test_critic_shapes() -> None:
    critic = SchedNetCritic(n_agents=3, obs_dim=5, hidden_dim=32)
    value, schedule_value = critic(torch.randn(6, 3, 5), torch.rand(6, 3))
    assert tuple(value.shape) == (6,)
    assert tuple(schedule_value.shape) == (6,)


def test_released_network_topology_is_independent_per_agent() -> None:
    agent = SchedNetAgent(n_agents=3, obs_dim=5, action_dim=4, bandwidth=1)

    assert len(agent.weight_generator.networks) == 3
    assert len(agent.message_encoder.networks) == 3
    assert len(agent.action_selector.networks) == 3
    assert agent.weight_generator.networks[0][0].weight.data_ptr() != (
        agent.weight_generator.networks[1][0].weight.data_ptr()
    )
    assert len(
        [layer for layer in agent.weight_generator.networks[0] if isinstance(layer, torch.nn.Linear)]
    ) == 3
    assert len(
        [layer for layer in agent.message_encoder.networks[0] if isinstance(layer, torch.nn.Linear)]
    ) == 2
    assert len(
        [layer for layer in agent.action_selector.networks[0] if isinstance(layer, torch.nn.Linear)]
    ) == 4
    assert agent.critic.shared_2.in_features == 64
    assert agent.critic.schedule_hidden.in_features == 64 + 3


def test_top_k_schedule_selects_highest_weights() -> None:
    weights = torch.tensor([[0.1, 0.9, 0.5, 0.2], [0.7, 0.3, 0.8, 0.6]])
    mask = top_k_schedule(weights, 2)
    # Row 0 top-2: idx 1 (0.9), idx 2 (0.5); row 1 top-2: idx 2 (0.8), idx 0 (0.7).
    assert torch.equal(mask, torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]))
    assert torch.equal(mask.sum(dim=-1), torch.full((2,), 2.0))
    # k larger than n_agents is clamped: everyone scheduled.
    assert torch.all(top_k_schedule(weights, 10) == 1.0)


def test_aggregate_broadcast_compacts_exact_bandwidth() -> None:
    messages = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    schedule = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    broadcast = aggregate_broadcast(messages, schedule)
    assert tuple(broadcast.shape) == (2, 3, 8)
    # Every agent hears the same broadcast.
    assert torch.equal(broadcast[:, 0], broadcast[:, 1])
    assert torch.equal(broadcast[:, 1], broadcast[:, 2])
    expected = torch.stack([messages[0, [0, 2]], messages[1, [1, 2]]])
    assert torch.allclose(broadcast[:, 0], expected.reshape(2, 8))


def test_aggregate_broadcast_is_differentiable_through_messages() -> None:
    # Messages are trained through the broadcast, so gradient must flow to scheduled agents only.
    messages = torch.randn(2, 3, 4, requires_grad=True)
    schedule = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    aggregate_broadcast(messages, schedule).sum().backward()
    assert messages.grad is not None
    assert torch.all(messages.grad[0, 1] == 0.0)   # agent 1 unscheduled in row 0
    assert torch.all(messages.grad[0, 0] == 3.0)   # scheduled: n_agents copies of the message


def test_schednet_agent_act_shapes() -> None:
    agent = SchedNetAgent(n_agents=3, obs_dim=5, action_dim=4, message_dim=2)
    obs = torch.randn(2, 3, 5)
    action, log_prob, weights, schedule = agent.act(obs, k=1)
    assert tuple(action.shape) == (2, 3)
    assert tuple(log_prob.shape) == (2, 3)
    assert tuple(weights.shape) == (2, 3)
    assert tuple(schedule.shape) == (2, 3)
    assert torch.all(weights >= 0.0) and torch.all(weights <= 1.0)
    assert torch.equal(schedule.sum(dim=-1), torch.ones(2))   # exactly k=1 agent scheduled per row


def test_schednet_agent_soft_update_runs() -> None:
    agent = SchedNetAgent(n_agents=3, obs_dim=5, action_dim=4, message_dim=2)
    assert not any(parameter.requires_grad for parameter in agent.target_critic.parameters())
    assert not any(
        parameter.requires_grad for parameter in agent.target_weight_generator.parameters()
    )
    agent.soft_update(0.01)


def _make_batch(batch_size: int = 4, n_agents: int = 3, obs_dim: int = 5, num_actions: int = 4):
    from examples.train_schednet import SchedNetBatch

    return SchedNetBatch(
        obs=torch.randn(batch_size, n_agents, obs_dim),
        actions=torch.randint(0, num_actions, (batch_size, n_agents)),
        rewards=torch.randn(batch_size),
        next_obs=torch.randn(batch_size, n_agents, obs_dim),
        dones=torch.zeros(batch_size),
        weights=torch.rand(batch_size, n_agents),
    )


class _ConstantWeightCritic(torch.nn.Module):
    """Critic whose schedule head is constant in the weights: zero gradient flows back, so the
    weight generator can only change if some other loss leaks into it."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs_all: torch.Tensor, weights_all: torch.Tensor):
        constant = weights_all.sum(dim=-1) * 0.0 + self.dummy.expand(obs_all.shape[0])
        return constant, constant


def test_schednet_update_action_loss_does_not_train_weight_generator() -> None:
    import copy
    from itertools import chain

    from examples.train_schednet import _update

    torch.manual_seed(0)
    agent = SchedNetAgent(
        n_agents=3, obs_dim=5, action_dim=4, message_dim=2, bandwidth=2,
    )
    # Stub out the weight-critic path: with a constant critic the weight-level loss has zero
    # gradient, so any change to the weight generator would come from the action-level loss —
    # which must not happen because the schedule is detached there.
    agent.critic = _ConstantWeightCritic()
    agent.target_critic = copy.deepcopy(agent.critic)

    action_opt = torch.optim.Adam(
        chain(agent.message_encoder.parameters(), agent.action_selector.parameters()), lr=1e-2,
    )
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=1e-2)
    weight_opt = torch.optim.Adam(agent.weight_generator.parameters(), lr=1e-2)

    generator_before = copy.deepcopy(agent.weight_generator.state_dict())
    selector_before = copy.deepcopy(agent.action_selector.state_dict())

    _update(agent, action_opt, critic_opt, weight_opt, _make_batch(), 2)

    for name, param in agent.weight_generator.state_dict().items():
        assert torch.equal(param, generator_before[name]), f"weight_generator.{name} changed"
    assert any(
        not torch.equal(param, selector_before[name])
        for name, param in agent.action_selector.state_dict().items()
    ), "action selector should still train"


def test_schednet_update_schedules_exactly_k_speakers(monkeypatch) -> None:
    import examples.train_schednet as train_schednet
    from examples.train_schednet import _update

    torch.manual_seed(0)
    k = 2
    agent = SchedNetAgent(
        n_agents=3, obs_dim=5, action_dim=4, message_dim=2, bandwidth=k,
    )
    action_opt = torch.optim.Adam(agent.action_selector.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)
    weight_opt = torch.optim.Adam(agent.weight_generator.parameters(), lr=1e-3)

    schedules: list[torch.Tensor] = []

    def spy(weights: torch.Tensor, k_arg: int) -> torch.Tensor:
        schedule = top_k_schedule(weights, k_arg)
        schedules.append(schedule)
        return schedule

    monkeypatch.setattr(train_schednet, "top_k_schedule", spy)
    _update(agent, action_opt, critic_opt, weight_opt, _make_batch(), k)

    # Once, for the policy loss. The released critic's value head takes only the state
    # (`use_action_in_critic` is False), so the TD target needs no schedule at all.
    assert len(schedules) == 1
    for schedule in schedules:
        assert torch.equal(schedule.sum(dim=-1), torch.full((schedule.shape[0],), float(k)))


def test_schednet_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "schednet.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        message_dim=2,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "schednet"
    assert summary["env"] == "navigation"
    assert summary["n_agents"] == 3
    assert len(summary["random_evaluation"]["returns"]) == 2
    assert summary["communication_rate"] == pytest.approx(1 / 3)
    assert checkpoint.exists()


def test_schednet_defaults_match_the_paper_and_release() -> None:
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=2,
        episodes=0,
        seed=5,
        evaluation_episodes=1,
    )

    config = summary["config"]
    assert config["gamma"] == 0.9
    assert config["tau"] == 0.05
    assert config["actor_learning_rate"] == 1e-5
    assert config["weight_learning_rate"] == 1e-5
    assert config["critic_learning_rate"] == 1e-4
    assert config["actor_hidden_dim"] == 32
    assert config["critic_hidden_dim"] == 64
    assert config["scheduler_hidden_dim"] == 32
    assert config["message_dim"] == 2
    assert config["batch_size"] == 64
    assert config["buffer_size"] == 10_000
    assert config["warmup_steps"] == 640
    assert config["epsilon_start"] == 0.5
    assert config["epsilon_final"] == 0.1
    assert config["epsilon_decay_steps"] == 750_000


def test_schednet_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = tmp_path / "schednet.pt"
    train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=2,
        seed=5,
        message_dim=2,
        buffer_size=64,
        batch_size=4,
        warmup_steps=0,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    state_dict = torch.load(str(checkpoint), weights_only=True)
    assert len(state_dict) > 0
    # It is a real SchedNet checkpoint: both policy levels are present.
    assert any(key.startswith("weight_generator") for key in state_dict)
    assert any(key.startswith("action_selector") for key in state_dict)


def test_update_uses_the_released_stochastic_policy_gradient() -> None:
    # Faithfulness guard. Released `ac_network.py`:
    #     log_prob = log(sum(actions * a_onehot));  entropy = -sum(actions * log(actions))
    #     loss = sum(-(log_prob * td_errors + 0.01 * entropy))
    # i.e. paper Eq. (4) -- a stochastic policy gradient weighted by the TD error of the
    # STATE-VALUE head, not a deterministic ascent on a Q(s, a) critic. A regression to the
    # old MADDPG/Gumbel scaffold would have no log_prob and no entropy term.
    from itertools import chain

    import examples.train_schednet as train_schednet
    from examples.train_schednet import ENTROPY_COEF, _update

    torch.manual_seed(0)
    k = 2
    agent = SchedNetAgent(
        n_agents=3, obs_dim=5, action_dim=4, message_dim=2, bandwidth=k,
    )
    batch = _make_batch()

    # Equation (4) uses the factorized joint-policy log probability and joint entropy.
    with torch.no_grad():
        next_value, _ = agent.target_critic(
            batch.next_obs, agent.target_weight_generator(batch.next_obs),
        )
        value, _ = agent.critic(batch.obs, batch.weights)
        td_error = batch.rewards + train_schednet.GAMMA * (1.0 - batch.dones) * next_value - value

    schedule = train_schednet.top_k_schedule(batch.weights, k)
    distribution = agent.action_selector.distribution(
        batch.obs, agent.broadcast_for(batch.obs, schedule),
    )
    expected = -(
        distribution.log_prob(batch.actions.long()).sum(dim=-1) * td_error
        + ENTROPY_COEF * distribution.entropy().sum(dim=-1)
    ).mean()
    expected.backward()
    reference = [p.grad.clone() for p in agent.action_selector.parameters()]

    # A fresh agent with identical parameters, stepped through the real update.
    torch.manual_seed(0)
    replica = SchedNetAgent(
        n_agents=3, obs_dim=5, action_dim=4, message_dim=2, bandwidth=k,
    )
    actor_opt = torch.optim.Adam(
        chain(replica.message_encoder.parameters(), replica.action_selector.parameters()), lr=0.0,
    )
    critic_opt = torch.optim.Adam(replica.critic.parameters(), lr=0.0)
    weight_opt = torch.optim.Adam(replica.weight_generator.parameters(), lr=0.0)
    _update(replica, actor_opt, critic_opt, weight_opt, batch, k)

    for got, want in zip(replica.action_selector.parameters(), reference):
        torch.testing.assert_close(got.grad, want, rtol=0.0, atol=1e-6)

    assert ENTROPY_COEF == 0.01
    assert not hasattr(agent, "action_critic") and not hasattr(agent, "weight_critic")


def test_terminal_transition_does_not_bootstrap_either_critic_head() -> None:
    from itertools import chain

    from examples.train_schednet import _update

    class ScalarCritic(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(value))
            self.schedule = torch.nn.Parameter(torch.tensor(value))

        def forward(self, obs_all: torch.Tensor, weights_all: torch.Tensor):
            input_link = 0.0 * weights_all.sum(dim=-1)
            return self.value + input_link, self.schedule + input_link

    agent = SchedNetAgent(n_agents=3, obs_dim=5, action_dim=4, bandwidth=2)
    agent.critic = ScalarCritic(0.0)
    agent.target_critic = ScalarCritic(10.0).requires_grad_(False)
    batch = _make_batch()
    batch.rewards.fill_(2.0)
    batch.dones.fill_(1.0)
    actor_opt = torch.optim.Adam(
        chain(agent.message_encoder.parameters(), agent.action_selector.parameters()), lr=0.0,
    )
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=0.0)
    weight_opt = torch.optim.Adam(agent.weight_generator.parameters(), lr=0.0)

    _update(agent, actor_opt, critic_opt, weight_opt, batch, 2)

    torch.testing.assert_close(agent.critic.value.grad, torch.tensor(-4.0))
    torch.testing.assert_close(agent.critic.schedule.grad, torch.tensor(-4.0))


def test_scheduler_critic_gradient_is_evaluated_at_replayed_priorities() -> None:
    from itertools import chain

    from examples.train_schednet import _update

    class QuadraticCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(()))

        def forward(self, obs, weights):
            return self.offset.expand(obs.shape[0]), self.offset + weights.square().sum(-1)

    torch.manual_seed(12)
    agent = SchedNetAgent(n_agents=3, obs_dim=5, action_dim=4, bandwidth=2)
    agent.critic = QuadraticCritic()
    agent.target_critic = QuadraticCritic().requires_grad_(False)
    batch = _make_batch()
    batch.weights[:] = torch.tensor([0.1, 0.8, 0.2])
    predictions = agent.weight_generator(batch.obs)
    # dQ/dw = 2*w at the stored priorities, not at today's generator outputs.
    expected_loss = -(predictions * (2 * batch.weights)).sum(-1).mean()
    expected = torch.autograd.grad(expected_loss, tuple(agent.weight_generator.parameters()))
    actor_opt = torch.optim.Adam(
        chain(agent.message_encoder.parameters(), agent.action_selector.parameters()), lr=0.0,
    )
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=0.0)
    weight_opt = torch.optim.Adam(agent.weight_generator.parameters(), lr=0.0)
    _update(agent, actor_opt, critic_opt, weight_opt, batch, 2)
    for parameter, gradient in zip(agent.weight_generator.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0.0, atol=1e-7)
    assert batch.weights.grad is None
