"""Equation and integration tests for the complete HAPPO learner."""

from __future__ import annotations

import pytest
import torch

from modmarl.algorithms.happo import HAPPOAgent, HAPPOBatch, ValueNorm


def _batch(learner: HAPPOAgent, steps: int = 12) -> HAPPOBatch:
    obs = torch.randn(steps, learner.n_agents, learner.actors[0].network.input_norm.normalized_shape[0])
    states = obs.reshape(steps, -1)
    with torch.no_grad():
        pairs = [learner.act(row) for row in obs]
        actions = torch.stack([pair[0] for pair in pairs])
        log_probs = torch.stack([pair[1] for pair in pairs])
        values = learner.critic(states)
    return HAPPOBatch(
        obs, states, actions, log_probs, values, torch.linspace(-1, 1, steps),
        torch.linspace(-2, 2, steps), torch.ones(steps, learner.n_agents),
        torch.ones(steps, learner.n_agents, learner.action_dim, dtype=torch.bool),
    )


def test_actors_are_independent_and_critic_is_centralized() -> None:
    learner = HAPPOAgent(3, 5, 4)
    first = {id(parameter) for parameter in learner.actors[0].parameters()}
    assert first.isdisjoint(id(parameter) for parameter in learner.actors[1].parameters())
    assert learner.actors[0].network.fc1.in_features == 5
    assert learner.critic.network.fc1.in_features == 15


def test_official_mpe_architecture_and_initialization() -> None:
    learner = HAPPOAgent(3, 5, 4)
    assert learner.actors[0].network.fc1.out_features == 128
    assert learner.actors[0].network.fc2.out_features == 128
    assert torch.allclose(learner.actors[0].network.output.bias, torch.zeros(4))


def test_action_availability_is_respected() -> None:
    learner = HAPPOAgent(2, 3, 4)
    mask = torch.tensor([[False, False, True, False]]).expand(2, -1)
    actions = learner.act(torch.randn(2, 3), available_actions=mask)[0]
    assert torch.equal(actions, torch.full((2,), 2))


def test_value_norm_matches_official_debiased_equations() -> None:
    normalizer = ValueNorm(beta=0.9)
    normalizer.update(torch.tensor([1.0, 3.0]))
    mean, variance = normalizer.statistics()
    assert mean.item() == pytest.approx(2.0)
    assert variance.item() == pytest.approx(1.0)


def test_terminal_gae_does_not_bootstrap() -> None:
    learner = HAPPOAgent(2, 3, 2, gamma=0.9, gae_lambda=0.8)
    learner.value_normalizer.update(torch.tensor([-1.0, 1.0]))
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.5])
    advantages, _ = learner.compute_gae(rewards, values, torch.tensor(10.0), torch.tensor([1.0, 0.0]))
    assert advantages[-1].item() == pytest.approx(0.5)
    assert advantages[0].item() == pytest.approx(1.85 + 0.9 * 0.8 * 0.5)


def test_sequential_factor_is_product_of_updated_policy_ratios() -> None:
    torch.manual_seed(3)
    learner = HAPPOAgent(3, 4, 3, actor_lr=2e-3)
    batch = _batch(learner)
    metrics = learner.update(batch, actor_epochs=2, critic_epochs=1, order=torch.tensor([2, 0, 1]))
    expected = torch.ones(batch.obs.shape[0])
    with torch.no_grad():
        for index in metrics["order"]:
            new_log_probs = learner.actors[int(index)].evaluate_actions(
                batch.obs[:, index], batch.actions[:, index], batch.available_actions[:, index],
            )[0]
            expected *= (new_log_probs - batch.old_log_probs[:, index]).exp()
    assert torch.allclose(metrics["factor"], expected)


def test_update_changes_every_actor_critic_and_normalizer() -> None:
    torch.manual_seed(4)
    learner = HAPPOAgent(3, 4, 3)
    batch = _batch(learner)
    before = [[parameter.detach().clone() for parameter in actor.parameters()] for actor in learner.actors]
    critic_before = [parameter.detach().clone() for parameter in learner.critic.parameters()]
    learner.update(batch, actor_epochs=1, critic_epochs=1)
    assert all(any(not torch.equal(a, b) for a, b in zip(saved, actor.parameters()))
               for saved, actor in zip(before, learner.actors))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, learner.critic.parameters()))
    assert learner.value_normalizer.debiasing_term > 0


def test_actor_with_no_active_samples_is_not_updated() -> None:
    torch.manual_seed(6)
    learner = HAPPOAgent(2, 4, 3)
    batch = _batch(learner)
    masks = batch.active_masks.clone()
    masks[:, 1] = 0.0
    masked = HAPPOBatch(**{**batch.__dict__, "active_masks": masks})
    before = [parameter.detach().clone() for parameter in learner.actors[1].parameters()]
    learner.update(masked, actor_epochs=1, critic_epochs=1, order=torch.tensor([0, 1]))
    assert all(torch.equal(a, b) for a, b in zip(before, learner.actors[1].parameters()))


def test_training_smoke_saves_complete_checkpoint(tmp_path) -> None:
    pytest.importorskip("gymnasium")
    from examples.train_happo import train

    checkpoint = tmp_path / "happo.pt"
    result = train(episodes=3, rollout_episodes=2, horizon=3, n_agents=2,
                   hidden_dim=16, ppo_epochs=1, evaluation_episodes=1, checkpoint=str(checkpoint))
    saved = torch.load(checkpoint, weights_only=True)
    assert result["episodes"] == 3
    assert set(saved) == {"model", "actor_optimizers", "critic_optimizer", "episodes"}
