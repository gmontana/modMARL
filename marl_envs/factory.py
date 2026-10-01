from __future__ import annotations

import warnings

import numpy as np

from .core import TargetSignalingEnv
from .hallway import NDQHallwayEnv
from .mpe import (
    BridgeNavigationControlEnv,
    DynamicPackControlEnv,
    FormationControlEnv,
    HiddenGateBridgeControlEnv,
    HiddenGateBridgeV2ControlEnv,
    LineControlEnv,
    NavigationControlEnv,
    SimpleSpreadMPEEnv,
)

ENV_CHOICES = (
    "simple_spread",
    "simple_spread_pz",
    "navigation",
    "bridge_navigation",
    "hidden_gate_bridge",
    "hidden_gate_bridge_v2",
    "hidden_gate_bridge_v2_oracle",
    "formation",
    "line",
    "pack",
    "ndq_hallway",
    "maic_hallway",
    "target_signaling",
    "delayed_signaling",
)


def _load_simple_spread_module():
    try:
        from mpe2 import simple_spread_v3

        return simple_spread_v3
    except ImportError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The environment `pettingzoo\.mpe(\.simple_spread_v3)?`.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from pettingzoo.mpe import simple_spread_v3

        return simple_spread_v3


class PettingZooSimpleSpreadEnv:
    """Thin adapter from PettingZoo parallel_env to the local trainer contract."""

    def __init__(
        self,
        n_agents: int = 3,
        horizon: int = 25,
        seed: int | None = None,
        local_ratio: float = 0.5,
    ) -> None:
        self.n_agents = n_agents
        self.horizon = horizon
        self.local_ratio = local_ratio
        self.dist_threshold = 0.1

        simple_spread_v3 = _load_simple_spread_module()
        self._env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=local_ratio,
            max_cycles=horizon,
            continuous_actions=False,
        )
        if seed is not None:
            self._env.reset(seed=seed)
        self.agent_names = list(self._env.possible_agents)
        first_agent = self.agent_names[0]
        self.obs_dim = int(self._env.observation_space(first_agent).shape[0])
        self.num_actions = int(self._env.action_space(first_agent).n)
        self._zero_obs = np.zeros((self.obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        obs_dict, infos = self._env.reset(seed=seed)
        obs = self._stack_obs(obs_dict)
        info = self._aggregate_info(infos)
        mean_distance, success = self._compute_metrics()
        info.update({"mean_distance": mean_distance, "success": success})
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action.shape}")

        action_dict = {agent_name: int(action[i]) for i, agent_name in enumerate(self.agent_names)}
        obs_dict, rewards, terminations, truncations, infos = self._env.step(action_dict)
        obs = self._stack_obs(obs_dict)
        reward = self._aggregate_reward(rewards)
        terminated = all(bool(terminations.get(agent_name, False)) for agent_name in self.agent_names)
        truncated = all(bool(truncations.get(agent_name, False)) for agent_name in self.agent_names)
        mean_distance, success = self._compute_metrics()
        info = self._aggregate_info(infos)
        info.update({"mean_distance": mean_distance, "success": success})
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def _stack_obs(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack(
            [
                np.asarray(obs_dict.get(agent_name, self._zero_obs), dtype=np.float32)
                for agent_name in self.agent_names
            ],
            axis=0,
        )

    def _aggregate_reward(self, rewards: dict[str, float]) -> float:
        if not rewards:
            return 0.0
        return float(np.mean([float(rewards.get(agent_name, 0.0)) for agent_name in self.agent_names]))

    def _aggregate_info(self, infos: dict[str, dict]) -> dict:
        return {"agent_infos": {agent_name: infos.get(agent_name, {}) for agent_name in self.agent_names}}

    def _compute_metrics(self) -> tuple[float, bool]:
        world = self._env.unwrapped.world
        agent_positions = np.stack([agent.state.p_pos for agent in world.agents], axis=0)
        landmark_positions = np.stack([landmark.state.p_pos for landmark in world.landmarks], axis=0)
        dists = np.linalg.norm(agent_positions[:, None, :] - landmark_positions[None, :, :], axis=-1)
        min_dists = dists.min(axis=0)
        occupied = int(np.sum(min_dists < self.dist_threshold))
        return float(np.mean(min_dists)), occupied == len(world.landmarks)


def make_env(
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    *,
    arena_size: float = 1.0,
    bridge_observe_radius: float | None = None,
    bridge_corridor_half_height: float | None = None,
    bridge_wall_margin: float | None = None,
    bridge_relay_fraction: float | None = None,
    bridge_corridor_capacity: int | None = None,
):
    if env_name == "ndq_hallway":
        return NDQHallwayEnv(n_agents=n_agents, horizon=horizon, seed=seed)
    if env_name == "maic_hallway":
        if n_agents != 3:
            raise ValueError("the MAIC Hallway task requires exactly three agents")
        return NDQHallwayEnv(
            n_agents=3, horizon=20, seed=seed, state_numbers=(2, 6, 10), reward_win=10.0,
        )
    if env_name == "target_signaling":
        return TargetSignalingEnv(n_agents=n_agents, seed=seed)
    if env_name == "delayed_signaling":
        return TargetSignalingEnv(n_agents=n_agents, seed=seed, horizon=horizon)
    if env_name == "simple_spread":
        return SimpleSpreadMPEEnv(n_agents=n_agents, horizon=min(horizon, 25), arena_size=arena_size, seed=seed)
    if env_name == "simple_spread_pz":
        return PettingZooSimpleSpreadEnv(n_agents=n_agents, horizon=min(horizon, 25), seed=seed)
    if env_name == "navigation":
        return NavigationControlEnv(n_agents=n_agents, horizon=horizon, arena_size=arena_size, seed=seed)
    if env_name in {"bridge_navigation", "hidden_gate_bridge", "hidden_gate_bridge_v2", "hidden_gate_bridge_v2_oracle"}:
        kwargs = {}
        if bridge_observe_radius is not None:
            kwargs["observe_radius"] = bridge_observe_radius
        if bridge_corridor_half_height is not None:
            kwargs["corridor_half_height"] = bridge_corridor_half_height
        if bridge_wall_margin is not None:
            kwargs["wall_margin"] = bridge_wall_margin
        if bridge_relay_fraction is not None:
            kwargs["relay_fraction"] = bridge_relay_fraction
        if bridge_corridor_capacity is not None:
            kwargs["corridor_capacity"] = bridge_corridor_capacity
        env_cls = {
            "bridge_navigation": BridgeNavigationControlEnv,
            "hidden_gate_bridge": HiddenGateBridgeControlEnv,
            "hidden_gate_bridge_v2": HiddenGateBridgeV2ControlEnv,
            "hidden_gate_bridge_v2_oracle": HiddenGateBridgeV2ControlEnv,
        }[env_name]
        if env_name == "hidden_gate_bridge_v2_oracle":
            kwargs["oracle_gate_observation"] = True
        return env_cls(n_agents=n_agents, horizon=horizon, arena_size=arena_size, seed=seed, **kwargs)
    if env_name == "formation":
        return FormationControlEnv(n_agents=n_agents, horizon=max(horizon, 50), arena_size=arena_size, seed=seed)
    if env_name == "line":
        return LineControlEnv(n_agents=n_agents, horizon=max(horizon, 50), arena_size=arena_size, seed=seed)
    if env_name == "pack":
        return DynamicPackControlEnv(n_agents=n_agents, horizon=max(horizon, 50), arena_size=arena_size, seed=seed)
    raise ValueError(f"unknown env: {env_name}")
