"""Golden equations, recurrent training, and execution contracts for IWoL."""

from __future__ import annotations

import inspect
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from examples.train_iwol import SOURCE_REVISION, train
from modmarl.algorithms.iwol import (
    IWoLActionKind,
    IWoLAgent,
    IWoLCommunication,
    IWoLMode,
    IWoLRollout,
    IWoLScheduler,
    LocalObservationEncoder,
    MaskedMessageAttention,
    communication_rate,
    compose_communication_graph,
)

CURVE_DATA = Path(__file__).resolve().parents[1] / "figures" / "curve_data"


def _learner(mode: IWoLMode = IWoLMode.IMPLICIT, **kwargs) -> IWoLAgent:
    return IWoLAgent(
        2,
        6,
        3,
        state_dim=12,
        mode=mode,
        position_slice=(0, 2),
        hidden_dim=8,
        latent_dim=4,
        n_encoder_layers=1,
        scheduler_heads=1,
        communication_heads=1,
        communication_hops=1,
        chunk_length=2,
        **kwargs,
    )


def _collected_batch(learner: IWoLAgent, steps: int = 3):
    rollout = IWoLRollout(learner)
    actor_hidden, critic_hidden = learner.initial_state(torch.device("cpu"))
    masks = torch.ones(learner.n_agents)
    physical = torch.ones(learner.n_agents, learner.n_agents)
    generator = torch.Generator().manual_seed(13)
    for timestep in range(steps):
        obs = torch.arange(
            learner.n_agents * learner.obs_dim, dtype=torch.float32,
        ).view(learner.n_agents, learner.obs_dim) / 10 + timestep
        states = obs.flatten().repeat(learner.n_agents, 1)
        previous_actor, previous_critic = actor_hidden, critic_hidden
        step = learner.act(
            obs,
            actor_hidden,
            critic_hidden,
            masks,
            physical,
            generator=generator,
        )
        actor_hidden, critic_hidden = step.actor_hidden, step.critic_hidden
        rollout.add(
            obs=obs,
            states=states,
            physical_graph=physical,
            step=step,
            actor_hidden=previous_actor,
            critic_hidden=previous_critic,
            masks=masks,
            team_reward=1.0,
        )
    rollout.finish_episode(torch.zeros(learner.n_agents), torch.zeros(learner.n_agents))
    return rollout.batch()


def test_design_block_pins_the_public_release_and_source_corrections():
    import modmarl.algorithms.iwol as iwol

    assert SOURCE_REVISION in iwol.__doc__
    assert "pre-softmax" in iwol.__doc__          # released attention-mask defect
    assert "physical graph" in iwol.__doc__
    assert "Gumbel noise" in iwol.__doc__
    # The observation encoder follows the release, not the paper's attention prose.
    assert "Appendix C.2" in iwol.LocalObservationEncoder.__doc__


def test_scheduler_uses_receiver_rows_and_sender_columns():
    scheduler = IWoLScheduler(2, 2, n_heads=1)
    scheduler.feature_encoder = nn.Identity()
    with torch.no_grad():
        scheduler.receiver_weight.copy_(torch.tensor([[1.0, 0.0]]))
        scheduler.sender_weight.copy_(torch.tensor([[0.0, 1.0]]))
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    logits = scheduler.link_logits(features)

    expected = torch.tensor([[[3.0, 5.0], [5.0, 7.0]]])
    torch.testing.assert_close(logits[..., 0], expected)
    torch.testing.assert_close(logits[..., 1], torch.zeros_like(expected))


def test_scheduler_replays_noise_with_hard_straight_through_gradients():
    scheduler = IWoLScheduler(4, 3, n_heads=1)
    features = torch.randn(2, 3, 4, requires_grad=True)
    noise = torch.zeros(2, 3, 3, 2)

    first, returned_noise = scheduler(features, noise=noise)
    second, _ = scheduler(features, noise=returned_noise)

    torch.testing.assert_close(first.detach(), first.detach().round(), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(first, second)
    first.sum().backward()
    assert features.grad is not None
    assert features.grad.abs().sum() > 0


def test_scheduler_matches_released_mlpbase_and_temperature() -> None:
    scheduler = IWoLScheduler(8, 3)
    modules = list(scheduler.feature_encoder)

    assert sum(isinstance(module, nn.Linear) for module in modules) == 2
    assert sum(isinstance(module, nn.LayerNorm) for module in modules) == 3
    assert scheduler.temperature == pytest.approx(0.1)


def test_physical_graph_masks_links_but_never_self_information():
    learned = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    physical = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])

    graph = compose_communication_graph(learned, physical)

    torch.testing.assert_close(graph, torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]))
    assert communication_rate(graph).item() == pytest.approx(0.5)
    assert communication_rate(torch.ones(4, 1, 1)).item() == 0.0


def test_masked_attention_has_exact_zero_weights_and_no_peer_leak():
    attention = MaskedMessageAttention(hidden_dim=4, n_heads=2)
    hidden = torch.randn(1, 3, 4)
    identity = torch.eye(3).unsqueeze(0)

    message, weights = attention(hidden, identity)
    changed = hidden.clone()
    changed[:, 1:] += 100.0
    changed_message, _ = attention(changed, identity)

    off_diagonal = 1.0 - torch.eye(3)
    assert (weights * off_diagonal).count_nonzero() == 0
    torch.testing.assert_close(message[:, 0], changed_message[:, 0])


def test_communication_processor_runs_released_transformer_hops():
    processor = IWoLCommunication(8, 8, n_heads=2, n_hops=3)
    features = torch.randn(2, 4, 8)

    message, weights = processor(features, torch.ones(2, 4, 4))

    assert message.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)
    assert len(processor.blocks) == 3


def test_observation_recurrence_resets_hidden_state_with_masks():
    learner = _learner()
    obs = torch.randn(1, 2, 6)
    first = torch.randn(1, 2, 8)
    second = torch.randn(1, 2, 8)
    reset = torch.zeros(1, 2)

    features_a, hidden_a = learner.actor.encoder(obs, first, reset)
    features_b, hidden_b = learner.actor.encoder(obs, second, reset)

    torch.testing.assert_close(features_a, features_b)
    torch.testing.assert_close(hidden_a, hidden_b)


def test_implicit_actor_is_peer_independent_and_reconstructs_both_targets():
    learner = _learner(IWoLMode.IMPLICIT)
    obs = torch.randn(1, 2, 6)
    changed = obs.clone()
    changed[:, 1] += 50.0
    hidden = torch.zeros(1, 2, 8)
    masks = torch.ones(1, 2)
    physical = torch.ones(1, 2, 2)

    original = learner.actor(obs, hidden, masks, physical, None, deterministic_graph=True)
    perturbed = learner.actor(changed, hidden, masks, physical, None, deterministic_graph=True)

    torch.testing.assert_close(original.distribution.logits[:, 0], perturbed.distribution.logits[:, 0])
    assert original.world_prediction.shape == (1, 2, 12)
    assert original.interaction_prediction is not None
    assert original.interaction_prediction.shape == (1, 2, 8)
    torch.testing.assert_close(original.graph, torch.eye(2).unsqueeze(0))


def test_explicit_actor_obeys_physical_disconnection():
    learner = _learner(IWoLMode.EXPLICIT)
    obs = torch.randn(1, 2, 6)
    changed = obs.clone()
    changed[:, 1] += 50.0
    hidden = torch.zeros(1, 2, 8)
    masks = torch.ones(1, 2)
    identity = torch.eye(2).unsqueeze(0)

    original = learner.actor(obs, hidden, masks, identity, None, deterministic_graph=True)
    perturbed = learner.actor(changed, hidden, masks, identity, None, deterministic_graph=True)

    torch.testing.assert_close(original.distribution.logits[:, 0], perturbed.distribution.logits[:, 0])
    torch.testing.assert_close(original.graph, identity)
    assert original.interaction_prediction is None


def test_actor_and_critic_own_independent_communication_protocols():
    learner = _learner(IWoLMode.EXPLICIT)

    assert learner.actor.scheduler is not learner.critic.scheduler
    assert learner.actor.communication is not learner.critic.communication
    actor_parameters = {parameter.data_ptr() for parameter in learner.actor.parameters()}
    critic_parameters = {parameter.data_ptr() for parameter in learner.critic.parameters()}
    assert actor_parameters.isdisjoint(critic_parameters)


def test_continuous_policy_uses_state_independent_diagonal_gaussian():
    learner = _learner(action_kind=IWoLActionKind.CONTINUOUS)
    actor_hidden, critic_hidden = learner.initial_state(torch.device("cpu"))

    step = learner.act(
        torch.randn(2, 6),
        actor_hidden,
        critic_hidden,
        torch.ones(2),
        torch.ones(2, 2),
    )

    assert step.actions.shape == (2, 3)
    assert step.log_probs.shape == (2,)
    assert learner.actor.action_head.log_std is not None
    assert learner.actor.action_head.log_std.shape == (3,)


def test_behavior_gumbel_noise_reproduces_collected_log_probabilities():
    learner = _learner(IWoLMode.EXPLICIT)
    obs = torch.randn(2, 6)
    actor_hidden, critic_hidden = learner.initial_state(torch.device("cpu"))
    masks = torch.ones(2)
    physical = torch.ones(2, 2)
    step = learner.act(
        obs,
        actor_hidden,
        critic_hidden,
        masks,
        physical,
        generator=torch.Generator().manual_seed(19),
    )

    replay = learner.actor(
        obs.unsqueeze(0),
        actor_hidden.unsqueeze(0),
        masks.unsqueeze(0),
        physical.unsqueeze(0),
        None,
        noise=step.actor_noise.unsqueeze(0),
    )
    replay_log_probs = learner.actor.action_head.log_prob(
        replay.distribution, step.actions.unsqueeze(0),
    )[0]

    torch.testing.assert_close(replay.graph[0], step.actor_graph)
    torch.testing.assert_close(replay_log_probs, step.log_probs)


def test_joint_recurrent_rollout_pads_without_splitting_agents():
    learner = _learner()
    batch = _collected_batch(learner, steps=3)

    assert batch.obs.shape == (2, 2, 2, 6)
    assert batch.actor_hidden.shape == (2, 2, 8)
    torch.testing.assert_close(batch.valid[:, :, 0], torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
    assert batch.valid[:, :, 0].equal(batch.valid[:, :, 1])


def test_gae_matches_hand_computed_terminal_returns():
    learner = _learner(gamma=1.0, gae_lambda=1.0)
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.zeros_like(rewards)

    advantages, returns = learner.compute_gae(
        rewards, values, torch.zeros(2), torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
    )

    expected = torch.tensor([[3.0, 3.0], [2.0, 2.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_implicit_message_teacher_is_detached_from_actor_objective():
    learner = _learner(IWoLMode.IMPLICIT)
    batch = _collected_batch(learner)
    index = torch.arange(batch.obs.shape[0])

    unroll = learner._unroll(batch, index)
    assert not unroll.teacher_messages.requires_grad
    policy, _, entropy, world, interaction = learner.losses(batch, index)
    learner.actor_optimizer.zero_grad(set_to_none=True)
    learner.critic_optimizer.zero_grad(set_to_none=True)
    (policy - learner.entropy_coef * entropy + world + interaction).backward()

    assert any(parameter.grad is not None for parameter in learner.actor.parameters())
    assert all(parameter.grad is None for parameter in learner.critic.parameters())


def test_update_rejects_rollouts_without_active_transitions():
    learner = _learner()
    batch = _collected_batch(learner, steps=1)
    inactive = replace(batch, active_masks=torch.zeros_like(batch.active_masks))

    with pytest.raises(ValueError, match="active transition"):
        learner.update(inactive, epochs=1)


@pytest.mark.parametrize("mode", list(IWoLMode))
def test_batch_size_one_update_is_finite_and_updates_both_networks(mode):
    learner = _learner(mode)
    batch = _collected_batch(learner, steps=1)
    actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in learner.critic.parameters()]

    metrics = learner.update(batch, epochs=1, num_minibatches=1)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, learner.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, learner.critic.parameters()))


def test_paper_and_release_defaults_are_explicit():
    learner = IWoLAgent(2, 4, 3)
    actor_group = learner.actor_optimizer.param_groups[0]
    critic_group = learner.critic_optimizer.param_groups[0]

    assert learner.hidden_dim == 128
    assert learner.actor.latent.network[-1].out_features == 32
    assert len(learner.critic.communication.blocks) == 4
    assert learner.critic.communication.blocks[0].attention.n_heads == 4
    assert learner.critic.scheduler.negative_slope == pytest.approx(1.2)
    assert learner.critic.scheduler.temperature == pytest.approx(0.1)
    assert actor_group["lr"] == pytest.approx(3e-4)
    assert critic_group["lr"] == pytest.approx(3e-4)
    assert actor_group["eps"] == pytest.approx(1e-5)
    assert actor_group["weight_decay"] == 0.0
    assert learner.gamma == pytest.approx(0.99)
    assert learner.gae_lambda == pytest.approx(0.95)
    assert learner.chunk_length == 10
    assert inspect.signature(learner.update).parameters["epochs"].default == 15


def test_implicit_evaluation_is_message_free_and_checkpoint_is_complete(tmp_path):
    checkpoint = tmp_path / "iwol.pt"

    result = train(
        episodes=0,
        n_agents=2,
        horizon=2,
        seed=3,
        hidden_dim=8,
        latent_dim=4,
        communication_heads=1,
        communication_hops=1,
        evaluation_episodes=1,
        checkpoint=str(checkpoint),
    )

    payload = torch.load(checkpoint, weights_only=True)
    assert result["communication_rate"] == 0.0
    assert result["final_evaluation"]["communication_rates"] == [0.0]
    assert result["source_revision"] == SOURCE_REVISION
    assert set(payload) == {
        "model",
        "actor_optimizer",
        "critic_optimizer",
        "episodes",
        "config",
        "source_revision",
    }


def test_explicit_trainer_smoke_records_execution_links():
    result = train(
        episodes=1,
        n_agents=2,
        horizon=2,
        seed=5,
        mode="explicit",
        hidden_dim=8,
        latent_dim=4,
        communication_heads=1,
        communication_hops=1,
        rollout_episodes=1,
        ppo_epochs=1,
        chunk_length=2,
        evaluation_episodes=1,
    )

    assert len(result["returns"]) == 1
    assert 0.0 <= result["communication_rate"] <= 1.0
    assert 0.0 <= result["training_actor_communication_rate"] <= 1.0


@pytest.mark.parametrize("seed", [3, 5, 7])
def test_committed_implicit_curve_passes_the_seedwise_acceptance_rule(seed):
    payload = json.loads((CURVE_DATA / f"iwol_seed{seed}.json").read_text())
    final_return = statistics.mean(payload["final_evaluation"]["returns"])

    assert payload["source_revision"] == SOURCE_REVISION
    assert payload["config"]["mode"] == "implicit"
    assert payload["episodes"] == len(payload["returns"]) == 9984
    assert len(payload["initial_evaluation"]["returns"]) == 100
    assert len(payload["random_evaluation"]["returns"]) == 100
    assert len(payload["final_evaluation"]["returns"]) == 100
    assert payload["communication_rate"] == 0.0
    assert final_return >= payload["validation_criterion"]["minimum_final_mean_return"]


def test_masked_attention_matches_a_true_masked_softmax_not_the_release() -> None:
    # `transformer_comm.py:114-122` multiplies PRE-softmax logits by the 0/1 graph, so a
    # masked sender keeps exp(0)=1 weight. This is the defect the module corrects, and it
    # is invisible at the loss level -- only the weights reveal it.
    torch.manual_seed(11)
    batch, agents, width, heads = 3, 6, 16, 4
    attention = MaskedMessageAttention(width, heads).double()
    hidden = torch.randn(batch, agents, width, dtype=torch.float64)
    graph = (torch.rand(batch, agents, agents, dtype=torch.float64) < 0.5).double()
    graph[..., torch.arange(agents), torch.arange(agents)] = 1.0   # self links always on

    _, weights = attention(hidden, graph)

    head_width = width // heads
    q = attention.query(hidden).view(batch, agents, heads, head_width).transpose(1, 2)
    k = attention.key(hidden).view(batch, agents, heads, head_width).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)
    mask = graph.unsqueeze(1).bool()
    exact = scores.masked_fill(~mask, -torch.inf).softmax(dim=-1)
    released = scores.softmax(dim=-1) * graph.unsqueeze(1)   # upstream: no renormalisation

    torch.testing.assert_close(weights, exact, rtol=0.0, atol=1e-12)
    assert (weights * (~mask)).abs().max().item() == 0.0        # masked senders exactly zero
    assert (released - exact).abs().max().item() > 1e-3         # the defect is real, not cosmetic


def test_observation_encoder_matches_the_released_mlpbase_and_rnn_layer() -> None:
    # MLPBase = LayerNorm(obs) then 1 + layer_N blocks of Linear -> ReLU -> LayerNorm;
    # RNNLayer = GRU then a LayerNorm on the output, with the raw cell state carried on.
    torch.manual_seed(13)
    obs_dim, hidden_dim, layers = 7, 12, 2
    encoder = LocalObservationEncoder(obs_dim, hidden_dim, n_encoder_layers=layers).double()
    obs = torch.randn(2, 4, obs_dim, dtype=torch.float64)
    hidden = torch.randn(2, 4, hidden_dim, dtype=torch.float64)
    masks = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)

    features, next_hidden = encoder(obs, hidden, masks)

    modules = list(encoder.feature)
    assert isinstance(modules[0], nn.LayerNorm) and modules[0].normalized_shape == (obs_dim,)
    assert sum(isinstance(m, nn.Linear) for m in modules) == 1 + layers

    x = obs
    for module in modules:
        x = module(x)
    expected_hidden = encoder.recurrent(
        x.reshape(8, hidden_dim), (hidden * masks.unsqueeze(-1)).reshape(8, hidden_dim),
    ).view(2, 4, hidden_dim)

    torch.testing.assert_close(next_hidden, expected_hidden, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        features, encoder.output_norm(expected_hidden), rtol=0.0, atol=1e-12,
    )
    assert not torch.allclose(features, next_hidden)    # the output LayerNorm is applied


def test_positional_embedding_reads_only_its_slice() -> None:
    # `pos_embed: True` in every published config; upstream adds
    # `pos_encoder(obs[start:end])` to the encoded state before the layer norm.
    torch.manual_seed(17)
    agents, feature_dim = 4, 8
    comm = IWoLCommunication(feature_dim, feature_dim, n_heads=2, n_hops=1, position_dim=2)
    features = torch.randn(2, agents, feature_dim)
    graph = torch.ones(2, agents, agents)
    positions = torch.randn(2, agents, 2)

    baseline, _ = comm(features, graph, positions)
    moved, _ = comm(features, graph, positions + 1.0)
    without, _ = comm(features, graph, None)

    assert not torch.allclose(baseline, moved)          # the slice reaches the message
    assert not torch.allclose(baseline, without)        # and is actually being added

    disabled = IWoLCommunication(feature_dim, feature_dim, n_heads=2, n_hops=1)
    assert disabled.pos_encoder is None
    torch.testing.assert_close(
        disabled(features, graph, positions)[0], disabled(features, graph, None)[0],
    )
