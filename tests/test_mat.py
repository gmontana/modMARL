"""Golden contracts for the released discrete MAT backbone and PPO learner."""

from __future__ import annotations

import torch

from modmarl.algorithms.mat import MATAgent, MATBackbone, MATBatch, MATRollout


def _batch(learner: MATAgent, samples: int = 4) -> MATBatch:
    obs = torch.randn(samples, learner.n_agents, learner.backbone.obs_dim)
    with torch.no_grad():
        actions, log_probs, values = learner.act(obs)
    return MATBatch(
        obs=obs,
        actions=actions,
        old_log_probs=log_probs,
        old_values=values,
        advantages=torch.randn_like(values),
        returns=torch.randn_like(values),
        active_masks=torch.ones_like(values),
        available_actions=torch.ones(
            samples, learner.n_agents, learner.action_dim, dtype=torch.bool,
        ),
    )


def test_shifted_actions_use_bos_then_preceding_one_hot():
    model = MATBackbone(3, 4, 3, embedding_dim=8)
    shifted = model.shifted_actions(torch.tensor([[2, 0, 1]]))
    expected = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]]
    )
    torch.testing.assert_close(shifted, expected)


def test_teacher_forcing_matches_autoregressive_log_probabilities():
    torch.manual_seed(3)
    model = MATBackbone(3, 5, 4, embedding_dim=16, n_heads=2).eval()
    obs = torch.randn(2, 3, 5)
    actions, sampled_log_probs, sampled_values = model.act(obs)
    parallel_log_probs, parallel_values, _ = model(obs, actions)
    torch.testing.assert_close(sampled_log_probs, parallel_log_probs)
    torch.testing.assert_close(sampled_values, parallel_values)


def test_decoder_is_causal_in_the_agent_action_order():
    torch.manual_seed(5)
    model = MATBackbone(3, 4, 3, embedding_dim=8).eval()
    obs = torch.randn(1, 3, 4)
    _, representation = model.encode(obs)
    first = model.decode(model.shifted_actions(torch.tensor([[0, 0, 0]])), representation)
    changed = model.decode(model.shifted_actions(torch.tensor([[0, 2, 1]])), representation)
    torch.testing.assert_close(first[:, :2], changed[:, :2])
    assert not torch.allclose(first[:, 2], changed[:, 2])


def test_encoder_is_noncausal_and_produces_one_value_per_agent():
    torch.manual_seed(7)
    model = MATBackbone(3, 4, 3, embedding_dim=8).eval()
    obs = torch.randn(1, 3, 4)
    changed = obs.clone()
    changed[:, 2, 0] += 4.0
    values, representation = model.encode(obs)
    changed_values, changed_representation = model.encode(changed)
    assert values.shape == (1, 3)
    assert representation.shape == (1, 3, 8)
    assert not torch.equal(values[:, 0], changed_values[:, 0])
    assert not torch.equal(representation[:, 0], changed_representation[:, 0])


def test_available_actions_apply_during_rollout_and_evaluation():
    model = MATBackbone(2, 3, 4, embedding_dim=8).eval()
    obs = torch.randn(1, 2, 3)
    available = torch.zeros(1, 2, 4, dtype=torch.bool)
    available[..., 2] = True
    actions, log_probs, _ = model.act(obs, available)
    assert actions.tolist() == [[2, 2]]
    evaluated, _, _ = model(obs, actions, available)
    torch.testing.assert_close(log_probs, evaluated)
    torch.testing.assert_close(evaluated, torch.zeros_like(evaluated))


def test_state_encoding_is_explicit_and_not_hard_coded_to_release_37_vector():
    model = MATBackbone(2, 3, 4, state_dim=6, encode_state=True, embedding_dim=8)
    obs = torch.randn(1, 2, 3)
    state = torch.randn(1, 2, 6)
    values, representation = model.encode(obs, state)
    assert values.shape == (1, 2)
    assert representation.shape == (1, 2, 8)


def test_rollout_preserves_joint_sequences_and_terminal_gae():
    learner = MATAgent(2, 3, 4, embedding_dim=8, gamma=1.0, gae_lambda=1.0)
    rollout = MATRollout(learner)
    for reward in (1.0, 2.0):
        rollout.add(
            obs=torch.zeros(2, 3),
            actions=torch.zeros(2, dtype=torch.long),
            log_probs=torch.zeros(2),
            values=torch.zeros(2),
            team_reward=reward,
        )
    rollout.finish_episode(torch.full((2,), 10.0), torch.zeros(2))
    batch = rollout.batch()
    assert batch.obs.shape == (2, 2, 3)
    torch.testing.assert_close(batch.returns, torch.tensor([[3.0, 3.0], [2.0, 2.0]]))


def test_truncation_bootstraps_the_final_value():
    learner = MATAgent(1, 2, 2, embedding_dim=8, gamma=1.0, gae_lambda=1.0)
    rollout = MATRollout(learner)
    rollout.add(
        obs=torch.zeros(1, 2),
        actions=torch.zeros(1, dtype=torch.long),
        log_probs=torch.zeros(1),
        values=torch.zeros(1),
        team_reward=2.0,
    )
    # Before the first ValueNorm update, normalized values use the 0.1 scale floor.
    rollout.finish_episode(torch.tensor([30.0]), torch.ones(1))
    torch.testing.assert_close(rollout.batch().returns, torch.tensor([[5.0]]))


def test_defaults_match_the_released_optimizer_and_ppo_schedule():
    learner = MATAgent(3, 4, 5)
    group = learner.optimizer.param_groups[0]
    assert group["lr"] == 5e-4
    assert group["eps"] == 1e-5
    assert group["weight_decay"] == 0.0
    assert learner.clip_epsilon == 0.2
    assert learner.entropy_coef == 0.01
    assert learner.max_grad_norm == 10.0
    assert learner.gamma == 0.99
    assert learner.gae_lambda == 0.95


def test_update_runs_fifteen_joint_ppo_epochs_by_default():
    torch.manual_seed(11)
    learner = MATAgent(2, 3, 4, embedding_dim=8)
    metrics = learner.update(_batch(learner, samples=2))
    assert metrics["model_updates"] == 15
    assert metrics["edge_updates"] == 0
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_batch_size_one_is_supported():
    learner = MATAgent(2, 3, 4, embedding_dim=8)
    metrics = learner.update(_batch(learner, samples=1), epochs=1)
    assert metrics["model_updates"] == 1


def test_checkpoint_round_trip_preserves_deterministic_joint_policy(tmp_path):
    torch.manual_seed(23)
    learner = MATAgent(2, 3, 4, embedding_dim=8).eval()
    obs = torch.randn(1, 2, 3)
    expected = learner.act(obs, deterministic=True)[0]
    checkpoint = tmp_path / "mat.pt"
    torch.save(learner.state_dict(), checkpoint)
    restored = MATAgent(2, 3, 4, embedding_dim=8).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.act(obs, deterministic=True)[0], expected)
