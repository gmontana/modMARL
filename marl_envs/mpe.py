from __future__ import annotations

import math
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.optimize import linear_sum_assignment


def assignment_min_dists(dists: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(dists)
    return dists[row_ind, col_ind]


class MPELikeTaskEnv(gym.Env[np.ndarray, np.ndarray], ABC):
    """Modernized, lightweight version of the MPE-style swarm tasks."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int,
        n_landmarks: int,
        horizon: int,
        arena_size: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.horizon = horizon
        self.arena_size = arena_size
        self.num_actions = 5
        self.damping = 0.25
        self.dt = 0.1
        self.force_scale = 5.0
        self.agent_size = 0.05
        self._rng = np.random.default_rng(seed)

        self.positions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.velocities = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.landmarks = np.zeros((self.n_landmarks, 2), dtype=np.float32)
        self.step_count = 0
        self.episode_return = 0.0
        self.last_mean_distance = 0.0
        self.last_success = False
        self.episode_collisions = 0
        self.episode_targets_caught = 0
        self.solve_step: int | None = None

        obs_dim = self._observation(0).shape[0]
        self.obs_dim = int(obs_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.full(self.n_agents, self.num_actions, dtype=np.int64))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.positions = self._rng.uniform(
            low=-self.arena_size,
            high=self.arena_size,
            size=(self.n_agents, 2),
        ).astype(np.float32)
        self.velocities = np.zeros_like(self.positions)
        self._reset_landmarks()
        self.step_count = 0
        self.episode_return = 0.0
        self.last_mean_distance = 0.0
        self.last_success = False
        self.episode_collisions = 0
        self.episode_targets_caught = 0
        self.solve_step = None
        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action.shape}")

        moves = np.array(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )[action]

        self.velocities = self.velocities * (1.0 - self.damping) + moves * self.force_scale * self.dt
        self.positions = self.positions + self.velocities * self.dt
        self.step_count += 1

        reward, mean_distance, success = self._task_reward()
        step_collisions = self._collision_count()
        self.episode_collisions += step_collisions
        self.last_mean_distance = mean_distance
        self.last_success = success
        self.episode_return += reward
        if success and self.solve_step is None:
            self.solve_step = self.step_count

        terminated = bool(success and self._terminate_on_success())
        truncated = self.step_count >= self.horizon and not terminated
        info = {
            "success": success if self._report_success_metric() else None,
            "mean_distance": mean_distance,
            "collisions": float(self.episode_collisions),
            "targets_caught": float(self.episode_targets_caught),
            "time_to_solve": self.solve_step if self._report_success_metric() else None,
            "episode_return": self.episode_return,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        return np.stack([self._observation(i) for i in range(self.n_agents)], axis=0).astype(np.float32)

    def _collision_count(self) -> int:
        collisions = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                distance = np.linalg.norm(self.positions[i] - self.positions[j])
                collisions += int(distance < 2.0 * self.agent_size)
        return collisions

    def _terminate_on_success(self) -> bool:
        return True

    def _report_success_metric(self) -> bool:
        return True

    @abstractmethod
    def _reset_landmarks(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _task_reward(self) -> tuple[float, float, bool]:
        raise NotImplementedError

    @abstractmethod
    def _observation(self, agent_idx: int) -> np.ndarray:
        raise NotImplementedError


class NavigationControlEnv(MPELikeTaskEnv):
    def __init__(self, n_agents: int = 3, horizon: int = 25, arena_size: float = 1.0, seed: int | None = None):
        self.dist_threshold = 0.1
        super().__init__(n_agents=n_agents, n_landmarks=n_agents, horizon=horizon, arena_size=arena_size, seed=seed)

    def _reset_landmarks(self) -> None:
        self.landmarks = self._rng.uniform(
            low=-self.arena_size,
            high=self.arena_size,
            size=(self.n_landmarks, 2),
        ).astype(np.float32)

    def _task_reward(self) -> tuple[float, float, bool]:
        dists = np.linalg.norm(self.positions[:, None, :] - self.landmarks[None, :, :], axis=-1)
        min_dists = assignment_min_dists(dists)
        reward = float(-np.mean(min_dists) - self._collision_count())
        mean_distance = float(np.mean(min_dists))
        success = bool(np.all(min_dists < self.dist_threshold))
        return reward, mean_distance, success

    def _observation(self, agent_idx: int) -> np.ndarray:
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]
        landmark_rel = (self.landmarks - pos).reshape(-1)
        others = np.delete(self.positions, agent_idx, axis=0) - pos
        return np.concatenate([vel, pos, landmark_rel, others.reshape(-1)])


class BridgeNavigationControlEnv(MPELikeTaskEnv):
    """Navigation with a wall and narrow corridor to induce relay/bottleneck structure."""

    def __init__(
        self,
        n_agents: int = 8,
        horizon: int = 75,
        arena_size: float = 1.0,
        seed: int | None = None,
        *,
        observe_radius: float | None = None,
        corridor_half_height: float | None = None,
        wall_margin: float | None = None,
        relay_fraction: float = 0.25,
        corridor_capacity: int = 2,
    ):
        self.dist_threshold = 0.12
        self.observe_radius = float(observe_radius if observe_radius is not None else 0.9 * arena_size)
        self.corridor_half_height = float(corridor_half_height if corridor_half_height is not None else 0.18 * arena_size)
        self.wall_margin = float(wall_margin if wall_margin is not None else 0.12 * arena_size)
        self.relay_fraction = float(relay_fraction)
        self.corridor_capacity = int(max(1, corridor_capacity))
        self.corridor_penalty = 0.25
        self.last_corridor_load = 0
        self.last_visible_landmarks_mean = 0.0
        super().__init__(n_agents=n_agents, n_landmarks=n_agents, horizon=horizon, arena_size=arena_size, seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self._reset_positions()
        self._reset_landmarks()
        self.step_count = 0
        self.episode_return = 0.0
        self.last_mean_distance = 0.0
        self.last_success = False
        self.episode_collisions = 0
        self.episode_targets_caught = 0
        self.solve_step = None
        self.last_corridor_load = 0
        self.last_visible_landmarks_mean = 0.0
        return self._observe(), {}

    def _reset_positions(self) -> None:
        relay_count = min(self.n_agents - 1, max(1, round(self.relay_fraction * self.n_agents)))
        left_count = self.n_agents - relay_count
        left_x = self._rng.uniform(
            low=-self.arena_size,
            high=-self.wall_margin,
            size=(left_count, 1),
        )
        left_y = self._rng.uniform(
            low=-self.arena_size,
            high=self.arena_size,
            size=(left_count, 1),
        )
        relay_x = self._rng.uniform(
            low=self.wall_margin,
            high=min(self.arena_size, self.wall_margin + 0.35 * self.arena_size),
            size=(relay_count, 1),
        )
        relay_y = self._rng.uniform(
            low=-self.corridor_half_height,
            high=self.corridor_half_height,
            size=(relay_count, 1),
        )
        self.positions = np.concatenate(
            [
                np.concatenate([left_x, left_y], axis=1),
                np.concatenate([relay_x, relay_y], axis=1),
            ],
            axis=0,
        ).astype(np.float32)
        self.velocities = np.zeros_like(self.positions)

    def _reset_landmarks(self) -> None:
        x = self._rng.uniform(
            low=self.wall_margin,
            high=self.arena_size,
            size=(self.n_landmarks, 1),
        )
        y = self._rng.uniform(
            low=-self.arena_size,
            high=self.arena_size,
            size=(self.n_landmarks, 1),
        )
        self.landmarks = np.concatenate([x, y], axis=1).astype(np.float32)

    def _corridor_mask(self, positions: np.ndarray) -> np.ndarray:
        return (np.abs(positions[:, 0]) <= self.wall_margin) & (np.abs(positions[:, 1]) <= self.corridor_half_height)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action.shape}")

        moves = np.array(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )[action]

        previous_positions = self.positions.copy()
        self.velocities = self.velocities * (1.0 - self.damping) + moves * self.force_scale * self.dt
        proposed_positions = self.positions + self.velocities * self.dt
        outside_corridor = np.abs(proposed_positions[:, 1]) > self.corridor_half_height
        blocked_from_left = outside_corridor & (previous_positions[:, 0] <= 0.0) & (proposed_positions[:, 0] > -self.wall_margin)
        blocked_from_right = outside_corridor & (previous_positions[:, 0] >= 0.0) & (proposed_positions[:, 0] < self.wall_margin)
        proposed_positions[blocked_from_left, 0] = -self.wall_margin
        proposed_positions[blocked_from_right, 0] = self.wall_margin
        self.velocities[blocked_from_left | blocked_from_right, 0] = 0.0
        proposed_positions[:, 0] = np.clip(proposed_positions[:, 0], -self.arena_size, self.arena_size)
        proposed_positions[:, 1] = np.clip(proposed_positions[:, 1], -self.arena_size, self.arena_size)
        self.positions = proposed_positions.astype(np.float32)
        self.step_count += 1

        reward, mean_distance, success = self._task_reward()
        step_collisions = self._collision_count()
        self.episode_collisions += step_collisions
        self.last_mean_distance = mean_distance
        self.last_success = success
        self.episode_return += reward
        if success and self.solve_step is None:
            self.solve_step = self.step_count

        terminated = bool(success and self._terminate_on_success())
        truncated = self.step_count >= self.horizon and not terminated
        corridor_load = int(self._corridor_mask(self.positions).sum())
        info = {
            "success": success,
            "mean_distance": mean_distance,
            "collisions": float(self.episode_collisions),
            "targets_caught": float(self.episode_targets_caught),
            "time_to_solve": self.solve_step,
            "episode_return": self.episode_return,
            "corridor_load": float(corridor_load),
            "visible_landmarks_mean": float(self.last_visible_landmarks_mean),
        }
        return self._observe(), float(reward), terminated, truncated, info

    def _task_reward(self) -> tuple[float, float, bool]:
        dists = np.linalg.norm(self.positions[:, None, :] - self.landmarks[None, :, :], axis=-1)
        min_dists = assignment_min_dists(dists)
        mean_distance = float(np.mean(min_dists))
        corridor_load = int(self._corridor_mask(self.positions).sum())
        corridor_penalty = self.corridor_penalty * max(0, corridor_load - self.corridor_capacity)
        reward = float(-np.mean(min_dists) - self._collision_count() - corridor_penalty)
        self.last_corridor_load = corridor_load
        success = bool(np.all(min_dists < self.dist_threshold))
        return reward, mean_distance, success

    def _observation(self, agent_idx: int) -> np.ndarray:
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]

        landmark_rel = self.landmarks - pos
        landmark_visible = np.linalg.norm(landmark_rel, axis=-1) <= self.observe_radius
        visible_landmark_rel = np.where(landmark_visible[:, None], landmark_rel, 0.0).reshape(-1)

        other_positions = np.delete(self.positions, agent_idx, axis=0)
        other_rel = other_positions - pos
        other_visible = np.linalg.norm(other_rel, axis=-1) <= self.observe_radius
        visible_other_rel = np.where(other_visible[:, None], other_rel, 0.0).reshape(-1)

        room_features = np.array(
            [
                float(pos[0] < -self.wall_margin),
                float(pos[0] > self.wall_margin),
                float(abs(pos[0]) <= self.wall_margin and abs(pos[1]) <= self.corridor_half_height),
            ],
            dtype=np.float32,
        )
        self.last_visible_landmarks_mean = float(np.mean(landmark_visible.astype(np.float32)))
        return np.concatenate([vel, pos, visible_landmark_rel.astype(np.float32), visible_other_rel.astype(np.float32), room_features])


class HiddenGateBridgeControlEnv(BridgeNavigationControlEnv):
    """Bridge task with a private gate bit that must cross the cut.

    The wall has two possible passages. Only agents on the right/relay side observe which
    passage is open; left-side agents must receive that information through communication
    to choose the correct lane.
    """

    def __init__(
        self,
        n_agents: int = 8,
        horizon: int = 75,
        arena_size: float = 1.0,
        seed: int | None = None,
        *,
        observe_radius: float | None = None,
        corridor_half_height: float | None = None,
        wall_margin: float | None = None,
        relay_fraction: float = 0.25,
        corridor_capacity: int = 2,
        gate_lane_center: float | None = None,
    ):
        self.active_gate = 1
        self.gate_lane_center = float(gate_lane_center if gate_lane_center is not None else 0.55 * arena_size)
        self.last_wrong_gate_count = 0
        self.last_gate_visible_mean = 0.0
        super().__init__(
            n_agents=n_agents,
            horizon=horizon,
            arena_size=arena_size,
            seed=seed,
            observe_radius=observe_radius if observe_radius is not None else 0.55 * arena_size,
            corridor_half_height=corridor_half_height,
            wall_margin=wall_margin,
            relay_fraction=relay_fraction,
            corridor_capacity=corridor_capacity,
        )
        self.wrong_gate_penalty = 0.75

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            # Keep the hidden gate tied to the episode seed for reproducible diagnostics.
            self._rng = np.random.default_rng(seed)
        self.active_gate = 1 if float(self._rng.uniform()) >= 0.5 else -1
        obs, info = super().reset(seed=None, options=options)
        info["gate_state"] = float(self.active_gate)
        return obs, info

    def _active_gate_center(self) -> float:
        return float(self.active_gate * self.gate_lane_center)

    def _gate_mask(self, positions: np.ndarray) -> np.ndarray:
        return (
            (np.abs(positions[:, 0]) <= self.wall_margin)
            & (np.abs(positions[:, 1] - self._active_gate_center()) <= self.corridor_half_height)
        )

    def _corridor_mask(self, positions: np.ndarray) -> np.ndarray:
        return self._gate_mask(positions)

    def _reset_positions(self) -> None:
        relay_count = min(self.n_agents - 1, max(1, round(self.relay_fraction * self.n_agents)))
        left_count = self.n_agents - relay_count
        left_x = self._rng.uniform(
            low=-self.arena_size,
            high=-self.wall_margin,
            size=(left_count, 1),
        )
        left_y = self._rng.uniform(
            low=-0.25 * self.arena_size,
            high=0.25 * self.arena_size,
            size=(left_count, 1),
        )
        relay_x = self._rng.uniform(
            low=self.wall_margin,
            high=min(self.arena_size, self.wall_margin + 0.35 * self.arena_size),
            size=(relay_count, 1),
        )
        relay_y = self._rng.uniform(
            low=self._active_gate_center() - self.corridor_half_height,
            high=self._active_gate_center() + self.corridor_half_height,
            size=(relay_count, 1),
        )
        self.positions = np.concatenate(
            [
                np.concatenate([left_x, left_y], axis=1),
                np.concatenate([relay_x, relay_y], axis=1),
            ],
            axis=0,
        ).astype(np.float32)
        self.positions[:, 1] = np.clip(self.positions[:, 1], -self.arena_size, self.arena_size)
        self.velocities = np.zeros_like(self.positions)

    def _reset_landmarks(self) -> None:
        x = self._rng.uniform(
            low=self.wall_margin,
            high=self.arena_size,
            size=(self.n_landmarks, 1),
        )
        y = self._rng.uniform(
            low=self._active_gate_center() - 0.35 * self.arena_size,
            high=self._active_gate_center() + 0.35 * self.arena_size,
            size=(self.n_landmarks, 1),
        )
        self.landmarks = np.concatenate([x, np.clip(y, -self.arena_size, self.arena_size)], axis=1).astype(np.float32)

    def _gate_visible(self, agent_idx: int) -> bool:
        pos = self.positions[agent_idx]
        return bool(pos[0] >= self.wall_margin or self._gate_mask(pos[None, :])[0])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action.shape}")

        moves = np.array(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )[action]

        previous_positions = self.positions.copy()
        self.velocities = self.velocities * (1.0 - self.damping) + moves * self.force_scale * self.dt
        proposed_positions = self.positions + self.velocities * self.dt
        in_active_gate = np.abs(proposed_positions[:, 1] - self._active_gate_center()) <= self.corridor_half_height
        attempting_cross = (previous_positions[:, 0] <= 0.0) & (proposed_positions[:, 0] > -self.wall_margin)
        attempting_cross |= (previous_positions[:, 0] >= 0.0) & (proposed_positions[:, 0] < self.wall_margin)
        blocked = attempting_cross & ~in_active_gate
        proposed_positions[blocked & (previous_positions[:, 0] <= 0.0), 0] = -self.wall_margin
        proposed_positions[blocked & (previous_positions[:, 0] >= 0.0), 0] = self.wall_margin
        self.velocities[blocked, 0] = 0.0
        proposed_positions[:, 0] = np.clip(proposed_positions[:, 0], -self.arena_size, self.arena_size)
        proposed_positions[:, 1] = np.clip(proposed_positions[:, 1], -self.arena_size, self.arena_size)
        self.positions = proposed_positions.astype(np.float32)
        self.last_wrong_gate_count = int(blocked.sum())
        self.step_count += 1

        reward, mean_distance, success = self._task_reward()
        step_collisions = self._collision_count()
        self.episode_collisions += step_collisions
        self.last_mean_distance = mean_distance
        self.last_success = success
        self.episode_return += reward
        if success and self.solve_step is None:
            self.solve_step = self.step_count

        terminated = bool(success and self._terminate_on_success())
        truncated = self.step_count >= self.horizon and not terminated
        corridor_load = int(self._corridor_mask(self.positions).sum())
        info = {
            "success": success,
            "mean_distance": mean_distance,
            "collisions": float(self.episode_collisions),
            "targets_caught": float(self.episode_targets_caught),
            "time_to_solve": self.solve_step,
            "episode_return": self.episode_return,
            "corridor_load": float(corridor_load),
            "visible_landmarks_mean": float(self.last_visible_landmarks_mean),
            "gate_state": float(self.active_gate),
            "wrong_gate_count": float(self.last_wrong_gate_count),
            "gate_visible_mean": float(self.last_gate_visible_mean),
        }
        return self._observe(), float(reward), terminated, truncated, info

    def _task_reward(self) -> tuple[float, float, bool]:
        dists = np.linalg.norm(self.positions[:, None, :] - self.landmarks[None, :, :], axis=-1)
        min_dists = assignment_min_dists(dists)
        mean_distance = float(np.mean(min_dists))
        corridor_load = int(self._corridor_mask(self.positions).sum())
        corridor_penalty = self.corridor_penalty * max(0, corridor_load - self.corridor_capacity)
        wrong_gate_penalty = self.wrong_gate_penalty * self.last_wrong_gate_count
        reward = float(-np.mean(min_dists) - self._collision_count() - corridor_penalty - wrong_gate_penalty)
        self.last_corridor_load = corridor_load
        success = bool(np.all(min_dists < self.dist_threshold))
        return reward, mean_distance, success

    def _observation(self, agent_idx: int) -> np.ndarray:
        base = super()._observation(agent_idx)
        gate_visible = self._gate_visible(agent_idx)
        gate_features = np.array(
            [
                float(gate_visible and self.active_gate > 0),
                float(gate_visible and self.active_gate < 0),
            ],
            dtype=np.float32,
        )
        self.last_gate_visible_mean = float(np.mean([self._gate_visible(i) for i in range(self.n_agents)]))
        return np.concatenate([base, gate_features])


class HiddenGateBridgeV2ControlEnv(HiddenGateBridgeControlEnv):
    """Stricter hidden-gate bridge with irreversible wrong-gate failures.

    This is a diagnostic information-bottleneck task: the acting side cannot cheaply infer
    the open lane by trial-and-error, while the oracle variant exposes the gate bit to all
    agents to verify that the task is solvable with the missing information.
    """

    def __init__(
        self,
        n_agents: int = 8,
        horizon: int = 75,
        arena_size: float = 1.0,
        seed: int | None = None,
        *,
        observe_radius: float | None = None,
        corridor_half_height: float | None = None,
        wall_margin: float | None = None,
        relay_fraction: float = 0.25,
        corridor_capacity: int = 2,
        gate_lane_center: float | None = None,
        oracle_gate_observation: bool = False,
    ):
        self.oracle_gate_observation = bool(oracle_gate_observation)
        self.fail_on_wrong_gate = True
        self.wrong_gate_failure = False
        super().__init__(
            n_agents=n_agents,
            horizon=horizon,
            arena_size=arena_size,
            seed=seed,
            observe_radius=observe_radius if observe_radius is not None else 0.45 * arena_size,
            corridor_half_height=corridor_half_height if corridor_half_height is not None else 0.10 * arena_size,
            wall_margin=wall_margin,
            relay_fraction=relay_fraction,
            corridor_capacity=corridor_capacity,
            gate_lane_center=gate_lane_center if gate_lane_center is not None else 0.62 * arena_size,
        )
        self.wrong_gate_penalty = 8.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        self.wrong_gate_failure = False
        obs, info = super().reset(seed=seed, options=options)
        info["oracle_gate_observation"] = bool(self.oracle_gate_observation)
        info["wrong_gate_failure"] = False
        return obs, info

    def _gate_visible(self, agent_idx: int) -> bool:
        if self.oracle_gate_observation:
            return True
        return super()._gate_visible(agent_idx)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        if self.fail_on_wrong_gate and self.last_wrong_gate_count > 0:
            self.wrong_gate_failure = True
            terminated = True
            truncated = False
            info["success"] = False
            info["wrong_gate_failure"] = True
            info["time_to_solve"] = None
        else:
            info["wrong_gate_failure"] = bool(self.wrong_gate_failure)
        info["oracle_gate_observation"] = bool(self.oracle_gate_observation)
        return obs, reward, terminated, truncated, info


class FormationControlEnv(MPELikeTaskEnv):
    def __init__(self, n_agents: int = 4, horizon: int = 50, arena_size: float = 1.0, seed: int | None = None):
        self.target_radius = 0.5
        self.dist_threshold = 0.05
        self.expected_positions = np.zeros((n_agents, 2), dtype=np.float32)
        super().__init__(n_agents=n_agents, n_landmarks=1, horizon=horizon, arena_size=arena_size, seed=seed)

    def _reset_landmarks(self) -> None:
        self.landmarks = self._rng.uniform(
            low=-0.5 * self.arena_size,
            high=0.5 * self.arena_size,
            size=(1, 2),
        ).astype(np.float32)

    def _task_reward(self) -> tuple[float, float, bool]:
        center = self.landmarks[0]
        relative_positions = self.positions - center
        thetas = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
        thetas = np.where(thetas < 0.0, thetas + 2.0 * math.pi, thetas)
        theta_min = float(np.min(thetas))
        ideal_sep = (2.0 * math.pi) / self.n_agents
        self.expected_positions = np.stack(
            [
                center
                + self.target_radius
                * np.array([math.cos(theta_min + i * ideal_sep), math.sin(theta_min + i * ideal_sep)])
                for i in range(self.n_agents)
            ],
            axis=0,
        ).astype(np.float32)
        dists = np.linalg.norm(self.positions[:, None, :] - self.expected_positions[None, :, :], axis=-1)
        min_dists = assignment_min_dists(dists)
        reward = float(-np.mean(np.clip(min_dists, 0.0, 2.0)))
        mean_distance = float(np.mean(min_dists))
        success = bool(np.all(min_dists < self.dist_threshold))
        return reward, mean_distance, success

    def _observation(self, agent_idx: int) -> np.ndarray:
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]
        landmark_rel = (self.landmarks[0] - pos).reshape(-1)
        return np.concatenate([vel, pos, landmark_rel])


class LineControlEnv(MPELikeTaskEnv):
    def __init__(self, n_agents: int = 4, horizon: int = 50, arena_size: float = 1.0, seed: int | None = None):
        self.total_sep = 1.25 * arena_size
        self.dist_threshold = 0.05
        self.expected_positions = np.zeros((n_agents, 2), dtype=np.float32)
        super().__init__(n_agents=n_agents, n_landmarks=2, horizon=horizon, arena_size=arena_size, seed=seed)

    def _reset_landmarks(self) -> None:
        start = self._rng.uniform(low=-0.25 * self.arena_size, high=0.25 * self.arena_size, size=(2,))
        theta = float(self._rng.uniform(0.0, 2.0 * math.pi))
        end = start + self.total_sep * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        while not (abs(end[0]) < self.arena_size and abs(end[1]) < self.arena_size):
            theta += math.radians(5.0)
            end = start + self.total_sep * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        self.landmarks = np.stack([start, end], axis=0).astype(np.float32)
        ideal_sep = self.total_sep / max(1, self.n_agents - 1)
        self.expected_positions = np.stack(
            [start + i * ideal_sep * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32) for i in range(self.n_agents)],
            axis=0,
        ).astype(np.float32)

    def _task_reward(self) -> tuple[float, float, bool]:
        dists = np.linalg.norm(self.positions[:, None, :] - self.expected_positions[None, :, :], axis=-1)
        min_dists = assignment_min_dists(dists)
        reward = float(-np.mean(np.clip(min_dists, 0.0, 2.0)))
        mean_distance = float(np.mean(min_dists))
        success = bool(np.all(min_dists < self.dist_threshold))
        return reward, mean_distance, success

    def _observation(self, agent_idx: int) -> np.ndarray:
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]
        landmark_rel = (self.landmarks - pos).reshape(-1)
        return np.concatenate([vel, pos, landmark_rel])


class DynamicPackControlEnv(MPELikeTaskEnv):
    def __init__(
        self,
        n_agents: int = 4,
        horizon: int = 50,
        arena_size: float = 1.0,
        seed: int | None = None,
        leader_count: int = 2,
    ) -> None:
        self.leader_count = min(leader_count, n_agents)
        self.catch_threshold = 0.12
        super().__init__(n_agents=n_agents, n_landmarks=1, horizon=horizon, arena_size=arena_size, seed=seed)

    def _reset_landmarks(self) -> None:
        self.landmarks = self._rng.uniform(
            low=-0.5 * self.arena_size,
            high=0.5 * self.arena_size,
            size=(1, 2),
        ).astype(np.float32)

    def _task_reward(self) -> tuple[float, float, bool]:
        distances = np.linalg.norm(self.positions - self.landmarks[0], axis=-1)
        mean_distance = float(np.mean(distances))
        success = bool(np.all(distances < self.catch_threshold))
        reward = -mean_distance
        if success:
            reward += 1.0
            self.episode_targets_caught += 1
            self._reset_landmarks()
        return reward, mean_distance, success

    def _terminate_on_success(self) -> bool:
        return False

    def _report_success_metric(self) -> bool:
        return False

    def _observation(self, agent_idx: int) -> np.ndarray:
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]
        target_rel = np.zeros(2, dtype=np.float32)
        is_leader = float(agent_idx < self.leader_count)
        if is_leader:
            target_rel = self.landmarks[0] - pos
        return np.concatenate([vel, pos, target_rel, np.array([is_leader], dtype=np.float32)])


class SimpleSpreadMPEEnv(MPELikeTaskEnv):
    """Stock simple_spread-style MPE task."""

    def __init__(self, n_agents: int = 3, horizon: int = 25, arena_size: float = 1.0, seed: int | None = None):
        self.dist_threshold = 0.1
        super().__init__(n_agents=n_agents, n_landmarks=n_agents, horizon=horizon, arena_size=arena_size, seed=seed)
        self.agent_size = 0.15

    def _reset_landmarks(self) -> None:
        self.landmarks = self._rng.uniform(
            low=-self.arena_size,
            high=self.arena_size,
            size=(self.n_landmarks, 2),
        ).astype(np.float32)

    def _is_collision(self, agent_i: int, agent_j: int) -> bool:
        if agent_i == agent_j:
            return False
        dist = np.linalg.norm(self.positions[agent_i] - self.positions[agent_j])
        return bool(dist < 2.0 * self.agent_size)

    def _task_reward(self) -> tuple[float, float, bool]:
        reward = 0.0
        occupied = 0
        min_dists: list[float] = []
        for landmark in self.landmarks:
            dists = np.linalg.norm(self.positions - landmark, axis=-1)
            min_dist = float(np.min(dists))
            min_dists.append(min_dist)
            reward -= min_dist
            occupied += int(min_dist < self.dist_threshold)
        collisions = 0
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if self._is_collision(i, j):
                    collisions += 1
        reward -= float(collisions)
        mean_distance = float(np.mean(min_dists))
        success = occupied == self.n_landmarks
        return reward, mean_distance, success

    def _observation(self, agent_idx: int) -> np.ndarray:
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]
        landmark_rel = (self.landmarks - pos).reshape(-1)
        others = np.delete(self.positions, agent_idx, axis=0) - pos
        return np.concatenate([vel, pos, landmark_rel, others.reshape(-1)])
