from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EpisodeInfo:
    episode_return: float
    success: bool
    mean_distance: float
    target: float


class LeaderFollowerTargetEnv(gym.Env[np.ndarray, np.ndarray]):
    """A simple cooperative 1D environment with partial observability.

    Agent 0 is the leader and observes the target location.
    All other agents are followers and do not observe the target directly.
    The team receives a shared reward based on how close every agent is to the target.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 3,
        horizon: int = 15,
        step_size: float = 0.15,
        world_radius: float = 1.0,
        target_tolerance: float = 0.12,
        seed: int | None = None,
    ) -> None:
        if n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        self.n_agents = n_agents
        self.horizon = horizon
        self.step_size = step_size
        self.world_radius = world_radius
        self.target_tolerance = target_tolerance
        self.obs_dim = 5
        self.num_actions = 3
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.full(self.n_agents, 3, dtype=np.int64))

        self.positions = np.zeros(self.n_agents, dtype=np.float32)
        self.target = np.float32(0.0)
        self.step_count = 0
        self.episode_return = 0.0

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
            low=-self.world_radius,
            high=self.world_radius,
            size=self.n_agents,
        ).astype(np.float32)
        self.target = np.float32(
            self._rng.uniform(low=-self.world_radius, high=self.world_radius)
        )
        self.step_count = 0
        self.episode_return = 0.0
        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action.shape}")

        delta = np.take(np.array([-self.step_size, 0.0, self.step_size], dtype=np.float32), action)
        self.positions = np.clip(
            self.positions + delta,
            -self.world_radius,
            self.world_radius,
        )
        self.step_count += 1

        distances = np.abs(self.positions - self.target)
        mean_distance = float(distances.mean())
        reward = -mean_distance
        success = bool(np.all(distances <= self.target_tolerance))
        if success:
            reward += 1.0

        self.episode_return += reward
        terminated = success
        truncated = self.step_count >= self.horizon and not terminated

        info = {
            "success": success,
            "mean_distance": mean_distance,
            "distances": distances.copy(),
            "target": float(self.target),
            "episode_return": self.episode_return,
        }
        return self._observe(), reward, terminated, truncated, info

    def _observe(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        obs[:, 0] = self.positions
        obs[:, 1] = np.float32(self.step_count / max(1, self.horizon - 1))
        obs[:, 2] = np.linspace(-1.0, 1.0, self.n_agents, dtype=np.float32)
        obs[0, 3] = 1.0
        obs[0, 4] = self.target / self.world_radius
        return obs


class TargetSignalingEnv(gym.Env[np.ndarray, np.ndarray]):
    """One-step cooperative signaling game.

    The leader observes a binary target bit and followers do not.
    The team is rewarded only when every agent outputs the correct bit.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_agents: int = 3, seed: int | None = None) -> None:
        if n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        self.n_agents = n_agents
        self.horizon = 1
        self.obs_dim = 5
        self.num_actions = 2
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.full(self.n_agents, self.num_actions, dtype=np.int64))

        self.target_bit = 0
        self.episode_return = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.target_bit = int(self._rng.integers(0, 2))
        self.episode_return = 0.0
        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action.shape}")
        correct_fraction = float(np.mean(action == self.target_bit))
        success = bool(correct_fraction == 1.0)
        reward = 2.0 * correct_fraction - 1.0
        self.episode_return += reward
        info = {
            "success": success,
            "mean_distance": 1.0 - correct_fraction,
            "target": int(self.target_bit),
            "episode_return": self.episode_return,
        }
        return self._observe(), reward, True, False, info

    def _observe(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        obs[:, 0] = np.linspace(-1.0, 1.0, self.n_agents, dtype=np.float32)
        obs[0, 1] = 1.0
        obs[0, 2 + self.target_bit] = 1.0
        return obs
