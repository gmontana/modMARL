from __future__ import annotations

import pytest
import torch

from modmarl.algorithms.ippo import IPPOAgent, IPPOBatch, IPPORollout, LocalValue
from modmarl.common.on_policy import compute_gae


def test_local_critic_is_shared_but_cannot_see_teammates() -> None:
    torch.manual_seed(0)
    critic = LocalValue(obs_dim=5, hidden_dims=(16, 8))
    obs = torch.randn(4, 3, 5)
    perturbed = obs.clone()
    perturbed[:, 1:] += 10.0
    assert torch.allclose(critic(obs)[:, 0], critic(perturbed)[:, 0])
    identical = obs[:, :1].expand(-1, 3, -1)
    assert torch.allclose(critic(identical)[:, 0], critic(identical)[:, 2])


def test_gae_matches_equation_four() -> None:
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.tensor([[0.5, 1.0], [0.25, 2.0]])
    bootstrap = torch.tensor([0.75, 0.0])
    actual, _ = compute_gae(rewards, values, bootstrap, gamma=0.9, gae_lambda=0.8)
    final_delta = rewards[1] + 0.9 * bootstrap - values[1]
    first_delta = rewards[0] + 0.9 * values[1] - values[0]
    expected = torch.stack((first_delta + 0.9 * 0.8 * final_delta, final_delta))
    assert torch.allclose(actual, expected)


def test_rollout_keeps_variable_episode_boundaries() -> None:
    rollout = IPPORollout(gamma=0.9, gae_lambda=0.8)
    for length in (2, 3):
        for _ in range(length):
            rollout.add(
                torch.randn(2, 4), torch.ones(2, 3, dtype=torch.bool),
                torch.zeros(2, dtype=torch.long), torch.zeros(2), torch.zeros(2), 1.0,
            )
        rollout.finish_episode(torch.zeros(2))
    batch = rollout.batch()
    assert batch.obs.shape == (10, 4)
    assert batch.available_actions.shape == (10, 3)
    assert batch.advantages.shape == (10,)


def test_value_clipping_matches_released_maximum_squared_error() -> None:
    agent = IPPOAgent(2, 2, hidden_dims=(4, 4), clip_epsilon=0.2, entropy_coef=0.0)
    for parameter in agent.parameters():
        torch.nn.init.zeros_(parameter)
    batch = IPPOBatch(
        obs=torch.zeros(1, 1, 2),
        available_actions=torch.ones(1, 1, 2, dtype=torch.bool),
        actions=torch.zeros(1, 1, dtype=torch.long),
        old_log_probs=torch.full((1, 1), -torch.log(torch.tensor(2.0))),
        old_values=torch.ones(1, 1),
        advantages=torch.ones(1, 1),
        returns=torch.full((1, 1), 2.0),
    )
    _, value_loss, _ = agent.losses(batch, torch.tensor([0]))
    # New V=0 has error 4; clipping it to old V-0.2 gives error 1.44, so PPO keeps 4.
    assert value_loss.item() == pytest.approx(4.0)


def test_policy_clipping_matches_equation_five() -> None:
    agent = IPPOAgent(1, 2, hidden_dims=(2, 2), clip_epsilon=0.2, entropy_coef=0.0)
    for parameter in agent.parameters():
        torch.nn.init.zeros_(parameter)
    batch = IPPOBatch(
        obs=torch.zeros(1, 1, 1),
        available_actions=torch.ones(1, 1, 2, dtype=torch.bool),
        actions=torch.zeros(1, 1, dtype=torch.long),
        old_log_probs=torch.log(torch.full((1, 1), 0.25)),
        old_values=torch.zeros(1, 1),
        advantages=torch.ones(1, 1),
        returns=torch.zeros(1, 1),
    )
    policy_loss, _, _ = agent.losses(batch, torch.tensor([0]))
    # Current probability .5 / old probability .25 = 2, clipped to 1.2.
    assert policy_loss.item() == pytest.approx(-1.2)


def test_action_mask_excludes_unavailable_actions() -> None:
    agent = IPPOAgent(3, 4, hidden_dims=(8, 4))
    obs = torch.randn(6, 3)
    available = torch.tensor([[True, False, False, False]]).expand(6, -1)
    actions, _, _ = agent.act(obs, available)
    assert torch.equal(actions, torch.zeros(6, dtype=torch.long))


def test_update_loss_replays_collection_action_mask() -> None:
    agent = IPPOAgent(1, 2, hidden_dims=(2, 2), entropy_coef=0.0)
    for parameter in agent.parameters():
        torch.nn.init.zeros_(parameter)
    batch = IPPOBatch(
        obs=torch.zeros(1, 1),
        available_actions=torch.tensor([[True, False]]),
        actions=torch.zeros(1, dtype=torch.long),
        old_log_probs=torch.zeros(1),
        old_values=torch.zeros(1),
        advantages=torch.ones(1),
        returns=torch.zeros(1),
    )
    policy_loss, _, entropy = agent.losses(batch, torch.tensor([0]))
    assert policy_loss.item() == pytest.approx(-1.0)
    assert entropy.item() == pytest.approx(0.0)


def test_update_changes_actor_and_local_critic() -> None:
    torch.manual_seed(2)
    agent = IPPOAgent(3, 2, hidden_dims=(8, 4))
    rollout = IPPORollout()
    for _ in range(5):
        obs = torch.randn(2, 3)
        available_actions = torch.ones(2, 2, dtype=torch.bool)
        actions, log_probs, values = agent.act(obs, available_actions)
        rollout.add(obs, available_actions, actions, log_probs, values, 1.0)
    rollout.finish_episode(torch.zeros(2))
    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
    metrics = agent.update(rollout, epochs=2, minibatch_size=3)
    assert any(not torch.equal(old, new) for old, new in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(critic_before, agent.critic.parameters()))
    assert set(metrics) == {"policy_loss", "value_loss", "entropy"}


def test_ippo_train_smoke_saves_optimizer(tmp_path) -> None:
    pytest.importorskip("gymnasium")
    from examples.train_ippo import train

    checkpoint = tmp_path / "ippo.pt"
    summary = train(
        env="navigation",
        n_agents=2,
        horizon=4,
        episodes=3,
        seed=5,
        hidden_dims=(16, 8),
        ppo_epochs=1,
        rollout_episodes=2,
        minibatch_size=4,
        checkpoint=str(checkpoint),
    )
    state = torch.load(checkpoint, weights_only=True)
    assert summary["episodes"] == 3
    assert {"model", "optimizer", "metadata"} == set(state)
    assert state["metadata"]["constructor"]["hidden_dims"] == (16, 8)
