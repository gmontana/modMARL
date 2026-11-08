"""DIAL/RIAL (Foerster et al., NeurIPS 2016) and the switch riddle against paper and release."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_dial import train
from marl_envs.switch import NONE, TELL, SwitchRiddleEnv
from modmarl.algorithms.dial import (
    DRU,
    EXECUTION_MESSAGE_MAGNITUDE,
    DIALAgent,
    DIALConfig,
    _EpisodeRecord,
)


def _agent(**overrides) -> DIALAgent:
    n_agents = overrides.pop("n_agents", 2)
    obs_dim = overrides.pop("obs_dim", 3)
    action_dim = overrides.pop("action_dim", 2)
    config = DIALConfig(**{"hidden_dim": 16, **overrides})
    return DIALAgent(n_agents, obs_dim, action_dim, config)


# ---------------------------------------------------------------- switch riddle
def test_switch_uses_paper_horizon() -> None:
    assert SwitchRiddleEnv(n_agents=3).horizon == 6
    assert SwitchRiddleEnv(n_agents=4).horizon == 10
    assert train(episodes=2, batch_size=2, evaluation_episodes=2, n_agents=4, horizon=99,
                 hidden_dim=8)["n_agents"] == 4


def test_switch_only_visitor_may_tell_and_observations_mark_the_room() -> None:
    env = SwitchRiddleEnv(n_agents=3, seed=0)
    obs, _ = env.reset(seed=3)
    visitor = int(env.visitors[0])
    assert obs[visitor].tolist() == [1.0, 0.0]
    assert all(obs[i].tolist() == [0.0, 1.0] for i in range(3) if i != visitor)
    available = env.available_actions()
    assert available[visitor].tolist() == [1.0, 1.0]
    assert all(available[i].tolist() == [1.0, 0.0] for i in range(3) if i != visitor)
    bad = np.zeros(3, dtype=np.int64)
    bad[(visitor + 1) % 3] = TELL
    with pytest.raises(ValueError):
        env.step(bad)


def _find_seed(predicate, n_agents=3):
    env = SwitchRiddleEnv(n_agents=n_agents)
    for seed in range(1000):
        env.reset(seed=seed)
        if predicate(env.visitors):
            return seed
    raise AssertionError("no seed found")


def test_switch_tell_reward_counts_visits_up_to_today() -> None:
    env = SwitchRiddleEnv(n_agents=3)
    seed = _find_seed(lambda v: len(set(v[:3].tolist())) == 3)
    env.reset(seed=seed)
    for _ in range(2):
        _, reward, done, _, _ = env.step(np.full(3, NONE))
        assert reward == 0.0 and not done
    actions = np.full(3, NONE)
    actions[env.visitors[2]] = TELL
    _, reward, done, _, info = env.step(actions)
    assert reward == 1.0 and done and info["success"]

    seed = _find_seed(lambda v: len(set(v[:2].tolist())) < 3)
    env.reset(seed=seed)
    actions = np.full(3, NONE)
    actions[env.visitors[0]] = TELL
    _, reward, done, _, info = env.step(actions)
    assert reward == -1.0 and done and not info["success"]


def test_switch_times_out_with_zero_reward() -> None:
    env = SwitchRiddleEnv(n_agents=3, seed=1)
    env.reset(seed=1)
    for day in range(env.horizon):
        _, reward, done, truncated, _ = env.step(np.full(3, NONE))
        assert reward == 0.0 and not truncated
        assert done == (day == env.horizon - 1)
    with pytest.raises(RuntimeError):
        env.step(np.full(3, NONE))


def test_switch_channel_links_consecutive_visitors_only() -> None:
    env = SwitchRiddleEnv(n_agents=3, seed=2)
    env.reset(seed=2)
    assert env.communication_mask().sum() == 0.0          # day 1 hears nothing
    assert env.message_senders()[env.visitors[0]] == 1.0
    assert env.message_senders().sum() == 1.0
    env.step(np.full(3, NONE))
    mask = env.communication_mask()
    assert mask.sum() == 1.0
    assert mask[env.visitors[1], env.visitors[0]] == 1.0  # includes self when re-sent (release)


def test_switch_is_seed_deterministic() -> None:
    first, second = SwitchRiddleEnv(3), SwitchRiddleEnv(3)
    first.reset(seed=5)
    second.reset(seed=5)
    assert np.array_equal(first.visitors, second.visitors)


# ------------------------------------------------------------------ architecture
def test_dru_adds_channel_noise_in_training_and_binarises_at_release_threshold() -> None:
    torch.manual_seed(0)
    dru = DRU(noise_std=2.0, threshold=0.5)
    message = torch.full((20_000, 1), 1.5)
    noisy = dru(message, discretize=False)
    assert abs(noisy.mean().item() - 1.5) < 0.05
    assert abs(noisy.std().item() - 2.0) < 0.05
    executed = dru(torch.tensor([[0.4], [0.6], [-3.0]]), discretize=True)
    assert executed.squeeze(-1).tolist() == [-20.0, 20.0, -20.0]
    assert torch.equal(DRU(0.0, 0.5)(message, discretize=False), message)
    assert torch.sigmoid(torch.tensor(EXECUTION_MESSAGE_MAGNITUDE)).item() > 0.999_999


def test_cnet_matches_section_6_1_architecture() -> None:
    agent = DIALAgent(3, 2, 2, DIALConfig())
    net = agent.networks[0]
    assert net.rnn.hidden_size == 128 and net.rnn.num_layers == 2
    assert all(parameter.abs().max() <= 0.08 for parameter in net.rnn.parameters())
    assert net.output[-1].out_features == 2 + 1                    # |U| + 1-bit message
    assert net.agent_lookup.num_embeddings == 3
    assert net.action_lookup.num_embeddings == 2 + 1               # index 0 = no previous action
    kinds = [type(layer).__name__ for layer in net.message_mlp]
    assert kinds == ["Sigmoid", "BatchNorm1d", "Linear", "ReLU"]


def test_rial_uses_one_hot_messages_without_logistic_or_batch_norm() -> None:
    agent = DIALAgent(3, 2, 2, DIALConfig(mode="rial"))
    net = agent.networks[0]
    assert [type(layer).__name__ for layer in net.message_mlp] == ["Linear"]
    assert net.output[-1].out_features == 2 + 2                    # 2**1 discrete messages
    assert net.message_action_lookup.num_embeddings == 2 + 1


def test_release_optimizer_and_parameter_sharing() -> None:
    shared, separate = _agent(), _agent(parameter_sharing=False)
    assert len(shared.networks) == 1 and len(separate.networks) == 2
    group = shared.optimizer.param_groups[0]
    assert isinstance(shared.optimizer, torch.optim.RMSprop)
    assert group["lr"] == 5e-4 and group["alpha"] == 0.99
    assert group["eps"] == 1e-8 and group["momentum"] == 0
    assert all(not parameter.requires_grad for parameter in shared.target_networks.parameters())


# ----------------------------------------------------------------------- channel
def test_routing_respects_mask_and_aggregation() -> None:
    sent = torch.tensor([[[1.0], [2.0], [4.0]]])                    # (B=1, senders=3, W=1)
    concat = _agent(n_agents=3)
    default = concat._mask(None, 1, sent.device)
    routed = concat._route(sent, default)
    assert routed[0, 0].tolist() == [0.0, 2.0, 4.0]                 # own message excluded
    assert routed[0, 2].tolist() == [1.0, 2.0, 0.0]
    summed = _agent(n_agents=3, message_aggregation="sum")
    mask = torch.zeros(1, 3, 3)
    mask[0, 1, 2] = 1.0
    out = summed._route(sent, mask)
    assert out[0, :, 0].tolist() == [0.0, 4.0, 0.0]


def _two_step_training(agent: DIALAgent) -> None:
    torch.manual_seed(1)
    agent.train()
    agent.begin_episode(4)
    obs = torch.randn(4, 2, 3)
    for _ in range(2):
        agent.act(obs)


def test_dial_backpropagates_receiver_error_into_sender() -> None:
    agent = _agent(parameter_sharing=False)
    _two_step_training(agent)
    receiver_q = agent._record.chosen_q[1][:, 1].sum()
    receiver_q.backward()
    sender_head = agent.networks[0].output[-1].weight.grad
    assert sender_head is not None
    assert sender_head[2:].abs().sum() > 0                          # message rows of agent 0


def test_rial_messages_carry_no_gradient_across_agents() -> None:
    agent = _agent(mode="rial", parameter_sharing=False)
    _two_step_training(agent)
    receiver_q = agent._record.chosen_q[1][:, 1].sum()
    receiver_q.backward()
    sender_head = agent.networks[0].output[-1].weight.grad
    assert sender_head is None or sender_head.abs().sum() == 0


def test_execution_channel_is_binary() -> None:
    agent = _agent()
    agent.eval()
    agent.begin_episode(3)
    agent.act(torch.randn(3, 2, 3), message_senders=torch.tensor([[1.0, 0.0]] * 3))
    sent = agent._state["sent"]
    assert set(sent[:, 0].flatten().tolist()) <= {-20.0, 20.0}
    assert torch.all(sent[:, 1] == 0.0)                             # silent agents send nothing


def test_unavailable_actions_are_never_chosen_even_when_exploring() -> None:
    agent = _agent(epsilon=1.0, action_dim=3)
    agent.train()
    agent.begin_episode(64)
    available = torch.zeros(64, 2, 3)
    available[..., 1] = 1.0
    actions = agent.act(torch.randn(64, 2, 3), available)
    assert torch.all(actions == 1)


def test_batch_norm_training_requires_parallel_episodes() -> None:
    agent = _agent()
    agent.train()
    with pytest.raises(ValueError):
        agent.begin_episode(1)
    agent.eval()
    agent.begin_episode(1)


# -------------------------------------------------------------------------- loss
def _record(**values) -> _EpisodeRecord:
    record = _EpisodeRecord()
    for key, items in values.items():
        setattr(record, key, [torch.tensor(item, dtype=torch.float32) for item in items])
    return record


def test_dqn_loss_bootstraps_from_target_and_stops_at_terminal() -> None:
    agent = _agent(gamma=0.9)
    record = _record(
        chosen_q=[[[1.0, 2.0]], [[0.5, 0.0]]],
        target_action_max=[[[9.0, 9.0]], [[3.0, 4.0]], [[100.0, 100.0]]],
        rewards=[[1.0], [2.0]], terminated=[[0.0], [1.0]], active=[[1.0], [1.0]],
    )
    loss, _, _ = agent._loss(record, steps=2)
    td_first = torch.tensor([1.0 + 0.9 * 3.0 - 1.0, 1.0 + 0.9 * 4.0 - 2.0])
    td_second = torch.tensor([2.0 - 0.5, 2.0 - 0.0])
    expected = 0.5 * (td_first.square().sum() + td_second.square().sum()) / (2 * 1)
    assert torch.isclose(loss, expected)


def test_finished_episodes_are_masked() -> None:
    agent = _agent()
    record = _record(
        chosen_q=[[[1.0, 1.0], [5.0, 5.0]]], target_action_max=[[[0.0, 0.0]] * 2, [[0.0, 0.0]] * 2],
        rewards=[[0.0, 0.0]], terminated=[[1.0, 1.0]], active=[[1.0, 0.0]],
    )
    loss, _, _ = agent._loss(record, steps=1)
    assert torch.isclose(loss, torch.tensor(0.5 * 2.0 / (2 * 2)))


def test_rial_averages_head_targets_like_the_release() -> None:
    agent = _agent(mode="rial", gamma=1.0)
    record = _record(
        chosen_q=[[[0.0, 0.0]]], chosen_message_q=[[[0.0, 0.0]]], message_taken=[[[1.0, 0.0]]],
        target_action_max=[[[0.0, 0.0]], [[2.0, 2.0]]],
        target_message_max=[[[0.0, 0.0]], [[4.0, 4.0]]],
        rewards=[[0.0]], terminated=[[0.0]], active=[[1.0]],
    )
    loss, _, _ = agent._loss(record, steps=1)
    # Agent 0 sent: both heads target (2 + 4) / 2 = 3. Agent 1 did not: action target 2, no message TD.
    expected = 0.5 * (3.0 ** 2 + 2.0 ** 2 + 3.0 ** 2) / 2
    assert torch.isclose(loss, torch.tensor(expected))


def test_truncated_batches_need_the_final_observation() -> None:
    agent = _agent()
    _two_step_training(agent)
    for _ in range(2):
        agent.observe(torch.zeros(4), torch.zeros(4), torch.ones(4))
    with pytest.raises(ValueError):
        agent.finish_episode()
    metrics = agent.finish_episode(torch.randn(4, 2, 3))
    assert np.isfinite(metrics["loss"])


def test_targets_refresh_every_interval_and_gradients_are_value_clipped(monkeypatch) -> None:
    agent = _agent(target_update_interval=2, learning_rate=1e-2)
    clipped = []
    original = torch.nn.utils.clip_grad_value_
    monkeypatch.setattr(torch.nn.utils, "clip_grad_value_",
                        lambda params, value: clipped.append(value) or original(params, value))

    def iteration() -> None:
        _two_step_training(agent)
        for _ in range(2):
            agent.observe(torch.ones(4), torch.zeros(4), torch.ones(4))
        agent.finish_episode(torch.randn(4, 2, 3))

    def same() -> bool:
        return all(torch.equal(a, b) for a, b in zip(
            agent.networks.state_dict().values(), agent.target_networks.state_dict().values()))

    iteration()
    assert not same()
    iteration()
    assert same()
    assert clipped == [10.0, 10.0]


# ----------------------------------------------------------------------- example
@pytest.mark.parametrize("mode", ["dial", "rial"])
def test_example_runs_switch_and_is_seed_reproducible(mode) -> None:
    kwargs = dict(mode=mode, episodes=8, batch_size=4, evaluation_episodes=4, seed=3, hidden_dim=16)
    first, second = train(**kwargs), train(**kwargs)
    assert first["iterations"] == 2 and first["episodes"] == 8
    assert first["returns"] == second["returns"]
    assert first["final_evaluation"]["returns"] == second["final_evaluation"]["returns"]
    assert 0.0 < first["oracle_mean_return"] <= 1.0


def test_example_runs_on_generic_broadcast_tasks() -> None:
    summary = train(env="target_signaling", episodes=4, batch_size=2, evaluation_episodes=2,
                    seed=1, hidden_dim=16, parameter_sharing=False)
    assert np.isfinite(summary["final_return"])
    assert "oracle_mean_return" not in summary
    navigation = train(env="navigation", episodes=2, batch_size=2, evaluation_episodes=2,
                       seed=1, hidden_dim=8)                  # default horizon must be usable
    assert np.isfinite(navigation["final_return"])


def test_training_task_draws_differ_between_nearby_run_seeds(monkeypatch) -> None:
    import examples.train_dial as runner

    drawn = {}
    original = runner.run_batch

    def spy(learner, envs, seeds, device, *, train, silence=False):
        if train:
            drawn.setdefault(spy.run_seed, []).append(list(seeds))
        return original(learner, envs, seeds, device, train=train, silence=silence)

    monkeypatch.setattr(runner, "run_batch", spy)
    for run_seed in (21, 23):
        spy.run_seed = run_seed
        runner.train(episodes=8, batch_size=4, evaluation_episodes=4, seed=run_seed, hidden_dim=8)
    overlap = {s for batch in drawn[21] for s in batch} & {s for batch in drawn[23] for s in batch}
    assert not overlap


@pytest.mark.slow
@pytest.mark.parametrize(("mode", "batches"), [("dial", 1000), ("rial", 3000)])
def test_switch_riddle_is_solved_through_the_channel(mode, batches) -> None:
    """Section 6.2, n = 3: reach the oracle and lose most of it when the channel is silenced.

    Development runs (one CPU thread, seed 21 pinned here; normalised greedy return, then
    with messages withheld): DIAL at 1,000 batches 1.0 -> 0.18 (seed 23 the same, ~45 s).
    RIAL at 1,000 batches 0.99 -> 0.54 on seed 21 and 1.0 -> 0.18 on seed 23; RIAL learns
    by trial and error and is slower (Section 6.2), and at 3,000 batches seed 21 gives
    0.99 -> 0.28 (~2 min). This is a regression check, not registered acceptance evidence.
    """
    summary = train(mode=mode, episodes=32 * batches, evaluation_episodes=512, seed=21)
    oracle = summary["oracle_mean_return"]
    final = summary["final_evaluation"]["mean_return"] / oracle
    silenced = summary["silenced_evaluation"]["mean_return"] / oracle
    assert final >= 0.95
    assert final - silenced >= 0.5
