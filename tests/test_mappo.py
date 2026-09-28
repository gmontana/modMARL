from __future__ import annotations

import pytest
import torch

from modmarl.algorithms.mappo import (
    MAPPOAgent,
    MAPPOBatch,
    MAPPORollout,
    ValueNorm,
    huber_loss,
)


def test_actor_is_local_while_critic_uses_centralized_state() -> None:
    learner = MAPPOAgent(3, 4, 5, state_dim=12, hidden_dim=8)
    actor_inputs = learner.actor.backbone.fc1.in_features
    critic_inputs = learner.critic.backbone.fc1.in_features
    assert actor_inputs == 4
    assert critic_inputs == 12


def test_parameter_sharing_gives_identical_outputs_for_identical_agents() -> None:
    torch.manual_seed(0)
    learner = MAPPOAgent(2, 3, 4, hidden_dim=8)
    hidden, critic_hidden = learner.initial_state(torch.device("cpu"))
    obs = torch.randn(1, 3).expand(2, -1)
    state = torch.randn(1, 6).expand(2, -1)
    actions, log_probs, values, _, _ = learner.act(
        obs, state, hidden, critic_hidden, torch.ones(2), deterministic=True,
    )
    assert actions[0] == actions[1]
    assert log_probs[0] == log_probs[1]
    assert values[0] == values[1]


def test_action_availability_mask_is_respected() -> None:
    learner = MAPPOAgent(2, 3, 4, hidden_dim=8)
    hidden, critic_hidden = learner.initial_state(torch.device("cpu"))
    actions = learner.act(
        torch.randn(2, 3), torch.randn(2, 6), hidden, critic_hidden, torch.ones(2),
        torch.tensor([[False, True, False, False]]).expand(2, -1),
    )[0]
    assert torch.equal(actions, torch.ones(2, dtype=torch.long))


def test_update_replays_action_masks_and_ignores_inactive_agents() -> None:
    torch.manual_seed(4)
    learner = MAPPOAgent(2, 3, 3, hidden_dim=8, chunk_length=2)
    batch = _one_chunk(learner)
    availability = torch.zeros_like(batch.available_actions)
    availability[..., 0] = True
    inactive = torch.zeros_like(batch.active_masks)
    constrained = MAPPOBatch(
        **{**batch.__dict__, "actions": torch.zeros_like(batch.actions),
           "available_actions": availability, "active_masks": inactive},
    )
    policy_loss, value_loss, entropy = learner.losses(
        constrained, torch.arange(constrained.obs.shape[0]),
    )
    assert policy_loss.item() == 0.0
    assert value_loss.item() == 0.0
    assert entropy.item() == 0.0


def test_recurrent_mask_resets_hidden_state() -> None:
    torch.manual_seed(1)
    learner = MAPPOAgent(1, 3, 2, hidden_dim=8)
    obs, state = torch.randn(1, 3), torch.randn(1, 3)
    nonzero = torch.randn(1, 8)
    first = learner.act(obs, state, nonzero, nonzero, torch.zeros(1), deterministic=True)
    second = learner.act(obs, state, torch.zeros_like(nonzero), torch.zeros_like(nonzero), torch.ones(1), deterministic=True)
    assert torch.allclose(first[3], second[3])
    assert torch.allclose(first[4], second[4])


def test_value_norm_matches_debiased_release_equations() -> None:
    normalizer = ValueNorm(beta=0.9)
    targets = torch.tensor([1.0, 3.0])
    normalizer.update(targets)
    mean, variance = normalizer.mean_variance()
    assert mean.item() == pytest.approx(2.0)
    assert variance.item() == pytest.approx(1.0)
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(targets)), targets)


def test_huber_loss_matches_supplement_table_four() -> None:
    error = torch.tensor([2.0, 20.0])
    assert torch.allclose(huber_loss(error, 10.0), torch.tensor([2.0, 150.0]))


def test_gae_uses_denormalized_values_and_terminal_mask() -> None:
    learner = MAPPOAgent(1, 2, 2, hidden_dim=4, gamma=0.9, gae_lambda=0.8)
    learner.value_normalizer.update(torch.tensor([0.0, 2.0]))
    values = learner.value_normalizer.normalize(torch.tensor([[0.5], [1.5]]))
    bootstrap = learner.value_normalizer.normalize(torch.tensor([2.0]))
    rewards = torch.tensor([[1.0], [2.0]])
    advantages, returns = learner.compute_gae(rewards, values, bootstrap, torch.tensor([[1.0], [0.0]]))
    final_delta = torch.tensor([0.5])
    first_delta = torch.tensor([1.85])
    expected = torch.stack((first_delta + 0.9 * 0.8 * final_delta, final_delta))
    assert torch.allclose(advantages, expected, atol=1e-5)
    assert torch.allclose(returns, expected + torch.tensor([[0.5], [1.5]]), atol=1e-5)


def test_rollout_splits_agents_and_pads_ten_step_chunks() -> None:
    learner = MAPPOAgent(2, 3, 2, hidden_dim=4, chunk_length=3)
    rollout = MAPPORollout(learner)
    actor_hidden, critic_hidden = learner.initial_state(torch.device("cpu"))
    for _ in range(5):
        obs, states = torch.randn(2, 3), torch.randn(2, 6)
        previous_actor, previous_critic = actor_hidden, critic_hidden
        actions, log_probs, values, actor_hidden, critic_hidden = learner.act(
            obs, states, actor_hidden, critic_hidden, torch.ones(2),
        )
        rollout.add(obs=obs, states=states, actions=actions, log_probs=log_probs, values=values,
                    actor_hidden=previous_actor, critic_hidden=previous_critic,
                    masks=torch.ones(2), team_reward=1.0)
    rollout.finish_episode(torch.zeros(2), torch.zeros(2))
    batch = rollout.batch()
    assert batch.obs.shape == (4, 3, 3)
    assert batch.valid.sum() == 10


def _one_chunk(learner: MAPPOAgent) -> MAPPOBatch:
    rollout = MAPPORollout(learner)
    actor_hidden, critic_hidden = learner.initial_state(torch.device("cpu"))
    for _ in range(3):
        obs, states = torch.randn(2, 3), torch.randn(2, 6)
        old_actor, old_critic = actor_hidden, critic_hidden
        actions, log_probs, values, actor_hidden, critic_hidden = learner.act(
            obs, states, actor_hidden, critic_hidden, torch.ones(2),
        )
        rollout.add(obs=obs, states=states, actions=actions, log_probs=log_probs, values=values,
                    actor_hidden=old_actor, critic_hidden=old_critic, masks=torch.ones(2), team_reward=1.0)
    rollout.finish_episode(torch.zeros(2), torch.zeros(2))
    return rollout.batch()


def test_full_update_changes_actor_critic_and_value_statistics() -> None:
    torch.manual_seed(3)
    learner = MAPPOAgent(2, 3, 2, hidden_dim=8, chunk_length=3)
    batch = _one_chunk(learner)
    actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in learner.critic.parameters()]
    metrics = learner.update(batch, epochs=2)
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, learner.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, learner.critic.parameters()))
    assert learner.value_normalizer.debiasing_term > 0
    assert set(metrics) == {"policy_loss", "value_loss", "entropy"}


def test_policy_surrogate_uses_official_clipped_probability_ratio() -> None:
    torch.manual_seed(5)
    learner = MAPPOAgent(2, 3, 2, hidden_dim=8, chunk_length=3, clip_epsilon=0.2)
    batch = _one_chunk(learner)
    index = torch.arange(batch.obs.shape[0])
    current_log_probs = learner._unroll(batch, index)[0].detach()
    ratio = 1.5
    controlled = MAPPOBatch(
        **{**batch.__dict__, "old_log_probs": current_log_probs - torch.log(torch.tensor(ratio)),
           "advantages": torch.ones_like(batch.advantages)},
    )
    policy_loss, _, _ = learner.losses(controlled, index)
    assert policy_loss.item() == pytest.approx(-1.2, abs=1e-6)


def test_train_smoke_saves_complete_checkpoint(tmp_path) -> None:
    pytest.importorskip("gymnasium")
    from examples.train_mappo import train

    checkpoint = tmp_path / "mappo.pt"
    summary = train(
        episodes=3, rollout_episodes=2, horizon=3, n_agents=2, hidden_dim=8,
        ppo_epochs=1, evaluation_episodes=1, checkpoint=str(checkpoint),
    )
    state = torch.load(checkpoint, weights_only=True)
    assert summary["episodes"] == 3
    assert set(state) == {"model", "actor_optimizer", "critic_optimizer", "episodes"}
