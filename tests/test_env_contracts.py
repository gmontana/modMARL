from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from marl_envs import MACPPEnv, NDQHallwayEnv, macpp_available, pettingzoo_simple_spread_available
from marl_envs.core import LeaderFollowerTargetEnv, TargetSignalingEnv
from marl_envs.factory import PettingZooSimpleSpreadEnv, make_env
from marl_envs.mpe import (
    BridgeNavigationControlEnv,
    DynamicPackControlEnv,
    FormationControlEnv,
    HiddenGateBridgeControlEnv,
    HiddenGateBridgeV2ControlEnv,
    NavigationControlEnv,
)


@pytest.mark.parametrize(
    ("env_factory", "action"),
    [
        (lambda: LeaderFollowerTargetEnv(n_agents=3, horizon=8, seed=11), np.array([0, 1, 2], dtype=np.int64)),
        (lambda: TargetSignalingEnv(n_agents=3, seed=13), np.array([1, 0, 1], dtype=np.int64)),
        (lambda: NavigationControlEnv(n_agents=3, horizon=6, seed=17), np.array([0, 1, 2], dtype=np.int64)),
        (lambda: NDQHallwayEnv(n_agents=2, seed=19), np.array([0, 1], dtype=np.int64)),
        (lambda: BridgeNavigationControlEnv(n_agents=6, horizon=6, seed=23), np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)),
        (lambda: HiddenGateBridgeControlEnv(n_agents=6, horizon=6, seed=29), np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)),
        (lambda: HiddenGateBridgeV2ControlEnv(n_agents=6, horizon=6, seed=31), np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)),
        (lambda: FormationControlEnv(n_agents=4, horizon=6, seed=19), np.array([0, 1, 2, 3], dtype=np.int64)),
    ],
)
def test_local_envs_are_seed_deterministic(env_factory, action: np.ndarray) -> None:
    env_a = env_factory()
    env_b = env_factory()
    try:
        obs_a, info_a = env_a.reset(seed=5)
        obs_b, info_b = env_b.reset(seed=5)
        assert np.allclose(obs_a, obs_b)
        assert info_a == info_b

        next_obs_a, reward_a, terminated_a, truncated_a, step_info_a = env_a.step(action)
        next_obs_b, reward_b, terminated_b, truncated_b, step_info_b = env_b.step(action)
        assert np.allclose(next_obs_a, next_obs_b)
        assert reward_a == reward_b
        assert terminated_a == terminated_b
        assert truncated_a == truncated_b
        assert step_info_a.keys() == step_info_b.keys()
        for key in ("success", "mean_distance"):
            assert step_info_a[key] == step_info_b[key]
    finally:
        close_a = getattr(env_a, "close", None)
        close_b = getattr(env_b, "close", None)
        if close_a is not None:
            close_a()
        if close_b is not None:
            close_b()


@pytest.mark.parametrize(
    "env",
    [
        LeaderFollowerTargetEnv(n_agents=3, horizon=8, seed=3),
        TargetSignalingEnv(n_agents=3, seed=5),
        NavigationControlEnv(n_agents=3, horizon=6, seed=7),
        NDQHallwayEnv(n_agents=2, seed=9),
        BridgeNavigationControlEnv(n_agents=6, horizon=6, seed=11),
        HiddenGateBridgeControlEnv(n_agents=6, horizon=6, seed=13),
        HiddenGateBridgeV2ControlEnv(n_agents=6, horizon=6, seed=17),
    ],
)
def test_local_envs_reject_invalid_action_shape(env) -> None:
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(np.zeros((env.n_agents, 1), dtype=np.int64))
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def test_factory_applies_expected_horizon_rules() -> None:
    navigation = make_env("navigation", n_agents=3, horizon=99, seed=0)
    hallway = make_env("ndq_hallway", n_agents=2, horizon=5, seed=0)
    bridge = make_env("bridge_navigation", n_agents=6, horizon=42, seed=0)
    hidden = make_env("hidden_gate_bridge", n_agents=6, horizon=43, seed=0)
    hidden_v2 = make_env("hidden_gate_bridge_v2", n_agents=6, horizon=44, seed=0)
    hidden_v2_oracle = make_env("hidden_gate_bridge_v2_oracle", n_agents=6, horizon=45, seed=0)
    formation = make_env("formation", n_agents=4, horizon=5, seed=0)
    line = make_env("line", n_agents=4, horizon=5, seed=0)
    try:
        assert navigation.horizon == 99  # no longer clamped; supports large-N scaling
        assert hallway.horizon == 16
        assert bridge.horizon == 42
        assert hidden.horizon == 43
        assert hidden_v2.horizon == 44
        assert hidden_v2_oracle.horizon == 45
        assert not hidden_v2.oracle_gate_observation
        assert hidden_v2_oracle.oracle_gate_observation
        assert formation.horizon == 50
        assert line.horizon == 50
    finally:
        navigation.close()
        hallway.close()
        bridge.close()
        hidden.close()
        hidden_v2.close()
        hidden_v2_oracle.close()
        formation.close()
        line.close()


def test_ndq_hallway_requires_simultaneous_arrival() -> None:
    env = NDQHallwayEnv(n_agents=2, seed=0, state_numbers=(6, 6))
    env.reset(seed=0)
    env.positions[:] = (1, 2)

    _, reward, terminated, truncated, info = env.step(np.array([1, 0]))

    assert terminated and not truncated
    assert reward == 0.0
    assert info["success"] is False


def test_ndq_hallway_matches_released_joint_win_rule() -> None:
    env = NDQHallwayEnv(n_agents=2, seed=0, state_numbers=(6, 6))
    env.reset(seed=0)
    env.positions[:] = (1, 1)

    _, reward, terminated, truncated, info = env.step(np.array([1, 1]))

    assert terminated and not truncated
    assert reward == 10.0
    assert info["success"] is True


def test_ndq_hallway_time_limit_is_terminal_like_release() -> None:
    env = NDQHallwayEnv(n_agents=2, horizon=16, seed=0, state_numbers=(6, 6))
    env.reset(seed=0)
    env.positions[:] = (6, 6)
    for _ in range(15):
        _, _, terminated, _, _ = env.step(np.array([0, 0]))
        assert not terminated

    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))

    assert terminated and not truncated
    assert reward == 0.0
    assert info["success"] is False


def test_dynamic_pack_contract_omits_success_metric() -> None:
    env = DynamicPackControlEnv(n_agents=4, horizon=6, seed=23)
    try:
        env.reset(seed=23)
        _, _, _, _, info = env.step(np.array([0, 1, 2, 3], dtype=np.int64))
        assert info["success"] is None
        assert info["time_to_solve"] is None
        assert "targets_caught" in info
    finally:
        env.close()


def test_bridge_navigation_blocks_wall_crossing_outside_corridor() -> None:
    env = BridgeNavigationControlEnv(n_agents=4, horizon=6, arena_size=1.0, seed=37)
    try:
        env.reset(seed=37)
        env.positions[0] = np.array([-0.13, 0.6], dtype=np.float32)
        env.velocities[0] = 0.0
        info = {}
        for _ in range(8):
            _, _, _, _, info = env.step(np.array([2, 0, 0, 0], dtype=np.int64))
        assert env.positions[0, 0] <= env.wall_margin + 1e-6
        assert "corridor_load" in info
    finally:
        env.close()


def test_bridge_navigation_factory_applies_custom_bottleneck_kwargs() -> None:
    env = make_env(
        "bridge_navigation",
        n_agents=6,
        horizon=20,
        seed=41,
        arena_size=1.5,
        bridge_observe_radius=1.1,
        bridge_corridor_half_height=0.07,
        bridge_wall_margin=0.2,
        bridge_relay_fraction=0.4,
        bridge_corridor_capacity=3,
    )
    try:
        assert isinstance(env, BridgeNavigationControlEnv)
        assert env.observe_radius == 1.1
        assert env.corridor_half_height == 0.07
        assert env.wall_margin == 0.2
        assert env.relay_fraction == 0.4
        assert env.corridor_capacity == 3
    finally:
        env.close()


def test_hidden_gate_bridge_exposes_private_gate_only_to_right_side() -> None:
    env = HiddenGateBridgeControlEnv(n_agents=6, horizon=10, arena_size=1.0, seed=43)
    try:
        obs, info = env.reset(seed=43)
        assert info["gate_state"] in {-1.0, 1.0}
        assert obs.shape == (6, env.obs_dim)
        left_gate_features = obs[env.positions[:, 0] < -env.wall_margin, -2:]
        right_gate_features = obs[env.positions[:, 0] >= env.wall_margin, -2:]
        assert np.allclose(left_gate_features, 0.0)
        assert np.all(np.sum(right_gate_features, axis=1) == 1.0)
    finally:
        env.close()


def test_hidden_gate_bridge_blocks_wrong_gate_crossing() -> None:
    env = HiddenGateBridgeControlEnv(n_agents=4, horizon=10, arena_size=1.0, seed=47)
    try:
        env.reset(seed=47)
        env.active_gate = 1
        env.positions[0] = np.array([-0.13, -env.gate_lane_center], dtype=np.float32)
        env.velocities[0] = 0.0
        info = {}
        for _ in range(8):
            _, _, _, _, info = env.step(np.array([2, 0, 0, 0], dtype=np.int64))
        assert env.positions[0, 0] <= env.wall_margin + 1e-6
        assert info["wrong_gate_count"] > 0.0
    finally:
        env.close()


def test_hidden_gate_bridge_v2_oracle_exposes_gate_to_left_side() -> None:
    env = HiddenGateBridgeV2ControlEnv(n_agents=6, horizon=10, arena_size=1.0, seed=53, oracle_gate_observation=True)
    try:
        obs, _ = env.reset(seed=53)
        left_gate_features = obs[env.positions[:, 0] < -env.wall_margin, -2:]
        assert left_gate_features.shape[0] > 0
        assert np.all(np.sum(left_gate_features, axis=1) == 1.0)
    finally:
        env.close()


def test_hidden_gate_bridge_v2_wrong_gate_is_terminal_failure() -> None:
    env = HiddenGateBridgeV2ControlEnv(n_agents=4, horizon=10, arena_size=1.0, seed=59)
    try:
        env.reset(seed=59)
        env.active_gate = 1
        env.positions[0] = np.array([-0.13, -env.gate_lane_center], dtype=np.float32)
        env.velocities[0] = 0.0
        _, _, terminated, truncated, info = env.step(np.array([2, 0, 0, 0], dtype=np.int64))
        assert terminated
        assert not truncated
        assert info["success"] is False
        assert info["wrong_gate_failure"] is True
    finally:
        env.close()


def test_maic_hallway_has_exact_paper_configuration() -> None:
    hallway = make_env("maic_hallway", n_agents=3, horizon=20, seed=0)
    assert hallway.state_numbers.tolist() == [2, 6, 10]
    assert hallway.horizon == 20
    assert hallway.reward_win == 10.0


def test_pettingzoo_simple_spread_wrapper_contract() -> None:
    if not pettingzoo_simple_spread_available():
        pytest.skip("PettingZoo simple_spread dependency is not installed")
    env = PettingZooSimpleSpreadEnv(n_agents=3, horizon=5, seed=29)
    try:
        obs, info = env.reset(seed=29)
        next_obs, reward, terminated, truncated, step_info = env.step(np.array([0, 1, 2], dtype=np.int64))
        assert obs.shape == (3, env.obs_dim)
        assert next_obs.shape == obs.shape
        assert isinstance(reward, float)
        assert "mean_distance" in info
        assert "mean_distance" in step_info
        assert not (terminated and truncated)
    finally:
        env.close()


def test_macpp_contract_when_available() -> None:
    if not macpp_available():
        pytest.skip("MACPP dependency is not installed")
    env = MACPPEnv(grid_size=5, n_agents=2, n_pickers=1, n_objects=1, horizon=5, seed=31)
    try:
        obs, info = env.reset(seed=31)
        graph_obs = env.graph_observation()
        assert obs.shape == (2, env.obs_dim)
        assert graph_obs.node_features.shape == (2, env.n_entities, env.node_feature_dim)
        assert graph_obs.relations.shape == (env.num_relations, env.n_entities, env.n_entities)
        assert "mean_distance" in info
    finally:
        env.close()
