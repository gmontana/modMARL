"""The learning diagnostic must not leak the answer to followers."""

import numpy as np

from marl_envs import make_env


def test_preparation_steps_reveal_neither_actions_nor_target():
    env = make_env("delayed_signaling", 3, 3, 0)
    followers = []
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        followers.append(obs[1:].copy())
        assert info == {}
        for _ in range(2):
            next_obs, reward, terminal, truncated, info = env.step(np.array([0, 1, 0]))
            np.testing.assert_array_equal(next_obs, obs)
            assert (reward, terminal, truncated, info) == (0.0, False, False, {})
        _, reward, terminal, truncated, info = env.step(np.full(3, env.target_bit))
        assert reward == 1.0 and terminal and not truncated and info["success"]
    for obs in followers:
        np.testing.assert_array_equal(obs, followers[0])


def test_constant_actions_succeed_on_exactly_one_of_two_targets():
    env = make_env("delayed_signaling", 3, 3, 0)
    for target in (0, 1):
        env.reset(seed=0)
        env.target_bit = target
        for _ in range(3):
            _, reward, terminal, _, info = env.step(np.zeros(3, dtype=int))
        assert terminal
        assert reward == (1.0 if target == 0 else -1.0)
        assert info["success"] == (target == 0)
