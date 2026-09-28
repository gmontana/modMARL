from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

MACPP_CHANNELS = 6
MACPP_NUM_ACTIONS = 6
MACPP_NUM_RELATIONS = 6
MACPP_NODE_FEATURE_DIM = 6


@dataclass(frozen=True)
class MACPPGraphObservation:
    node_features: np.ndarray
    relations: np.ndarray


def macpp_available() -> bool:
    return importlib.util.find_spec("gym") is not None and importlib.util.find_spec("macpp") is not None


def assignment_min_dists(dists: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(dists)
    return dists[row_ind, col_ind]


def _spatial_relations(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    deltas = positions[None, :, :] - positions[:, None, :]
    dx = deltas[..., 0]
    dy = deltas[..., 1]
    identity = np.eye(len(positions), dtype=bool)

    relations = np.zeros((MACPP_NUM_RELATIONS, len(positions), len(positions)), dtype=np.float32)
    relations[0] = (dx < 0.0) & ~identity  # left
    relations[1] = (dx > 0.0) & ~identity  # right
    relations[2] = (dy < 0.0) & ~identity  # top
    relations[3] = (dy > 0.0) & ~identity  # bottom
    relations[4] = (np.abs(dx) + np.abs(dy) == 1.0) & ~identity  # adjacent
    relations[5] = ((dx == 0.0) | (dy == 0.0)) & ~identity  # aligned
    return relations


class MACPPEnv:
    """Thin adapter around the public Collaborative Pick and Place environment."""

    def __init__(
        self,
        grid_size: int = 5,
        n_agents: int = 2,
        n_pickers: int = 1,
        n_objects: int = 1,
        horizon: int = 25,
        version: str = "v0",
        seed: int | None = None,
        debug_mode: bool = False,
    ) -> None:
        if not macpp_available():
            raise ImportError("MACPP support requires `gym` and the `macpp` package")

        import gym  # type: ignore
        import macpp  # noqa: F401  (the separate macpp package registers its gym env ids)

        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_pickers = n_pickers
        self.n_objects = n_objects
        self.horizon = horizon
        self.version = version
        self.obs_dim = grid_size * grid_size * MACPP_CHANNELS
        self.num_actions = MACPP_NUM_ACTIONS
        self.num_relations = MACPP_NUM_RELATIONS
        self.node_feature_dim = MACPP_NODE_FEATURE_DIM
        self.n_entities = n_agents + 2 * n_objects

        env_id = f"macpp-{grid_size}x{grid_size}-{n_agents}a-{n_pickers}p-{n_objects}o-{version}"
        self._env = gym.make(env_id, debug_mode=debug_mode, disable_env_checker=True)
        self._raw_obs: dict[str, Any] | None = None
        self.step_count = 0
        self.episode_return = 0.0
        self.solve_step: int | None = None
        if seed is not None:
            self.reset(seed=seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        raw_obs, _ = self._env.reset(seed=seed)
        self._raw_obs = raw_obs
        self.step_count = 0
        self.episode_return = 0.0
        self.solve_step = None
        flat_obs = self._encode_policy_obs()
        info = self._build_info(success=False)
        return flat_obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action.shape}")

        raw_obs, reward, terminated, info = self._env.step(action.tolist())
        self._raw_obs = raw_obs
        self.step_count += 1
        self.episode_return += float(reward)
        success = bool(terminated)
        if success and self.solve_step is None:
            self.solve_step = self.step_count
        truncated = self.step_count >= self.horizon and not success
        info_payload = self._build_info(success=success)
        info_payload.update(info)
        return self._encode_policy_obs(), float(reward), success, truncated, info_payload

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def graph_observation(self) -> MACPPGraphObservation:
        env = self._unwrapped_env()
        agent_positions = np.asarray([agent.position for agent in env.agents], dtype=np.float32)
        object_positions = np.asarray([obj.position for obj in env.objects], dtype=np.float32)
        goal_positions = np.asarray(env.goals, dtype=np.float32)
        positions = np.concatenate([agent_positions, object_positions, goal_positions], axis=0)

        node_features = np.zeros((self.n_agents, self.n_entities, self.node_feature_dim), dtype=np.float32)
        for ego_id, ego_features in enumerate(node_features):
            for agent_id, agent in enumerate(env.agents):
                ego_features[agent_id, 0] = 1.0
                ego_features[agent_id, 3] = float(agent.picker)
                ego_features[agent_id, 4] = float(agent.carrying_object is not None)
                ego_features[agent_id, 5] = float(agent_id == ego_id)
            for object_id, obj in enumerate(env.objects):
                offset = self.n_agents + object_id
                ego_features[offset, 1] = 1.0
                ego_features[offset, 4] = float(obj.carrying_agent is not None)
            for goal_id in range(self.n_objects):
                offset = self.n_agents + self.n_objects + goal_id
                ego_features[offset, 2] = 1.0

        relations = _spatial_relations(positions)
        return MACPPGraphObservation(node_features=node_features, relations=relations)

    def _encode_policy_obs(self) -> np.ndarray:
        env = self._unwrapped_env()
        grid_obs = np.zeros(
            (self.n_agents, self.grid_size, self.grid_size, MACPP_CHANNELS),
            dtype=np.float32,
        )

        for ego_id in range(self.n_agents):
            grid = grid_obs[ego_id]
            for agent_id, agent in enumerate(env.agents):
                x, y = agent.position
                grid[x, y, 0] = 1.0
                grid[x, y, 4] = 1.0 if agent.carrying_object is not None else -1.0
                grid[x, y, 5] = float(agent.picker)
                if agent_id == ego_id:
                    grid[x, y, 3] = 1.0
            for obj in env.objects:
                x, y = obj.position
                grid[x, y, 1] = 1.0
            for goal in env.goals:
                x, y = goal
                grid[x, y, 2] = 1.0
        return grid_obs.reshape(self.n_agents, -1)

    def _build_info(self, *, success: bool) -> dict[str, Any]:
        env = self._unwrapped_env()
        object_positions = np.asarray([obj.position for obj in env.objects], dtype=np.float32)
        goal_positions = np.asarray(env.goals, dtype=np.float32)
        if len(object_positions) == 0 or len(goal_positions) == 0:
            mean_distance = 0.0
            targets_caught = 0.0
        else:
            dists = np.abs(object_positions[:, None, :] - goal_positions[None, :, :]).sum(axis=-1)
            mean_distance = float(np.mean(assignment_min_dists(dists)))
            occupied_goals = 0
            for obj in env.objects:
                occupied_goals += int(tuple(obj.position) in set(env.goals) and obj.carrying_agent is None)
            targets_caught = float(occupied_goals)
        return {
            "success": success,
            "mean_distance": mean_distance,
            "collisions": 0.0,
            "targets_caught": targets_caught,
            "time_to_solve": self.solve_step,
            "episode_return": self.episode_return,
        }

    def _unwrapped_env(self):
        return getattr(self._env, "unwrapped", self._env)


__all__ = [
    "MACPPEnv",
    "MACPPGraphObservation",
    "macpp_available",
]
