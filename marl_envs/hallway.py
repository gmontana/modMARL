"""The didactic Hallway task released with NDQ.

Model: each agent occupies a private one-dimensional hallway and observes only
its own integer position. Actions wait, move toward zero, or move away from it.
Invariants: the team wins only when every agent reaches zero on the same step;
an episode fails immediately if only a strict subset reaches zero.
Interface: the standard local ``reset``/``step`` multi-agent environment API.
Why: NDQ's communication bottleneck was introduced and evaluated on this task,
so it provides a paper-matched validation setting without requiring StarCraft II.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class NDQHallwayEnv:
    """Faithful adapter of the official NDQ ``Join1Env`` hallway task."""

    def __init__(
        self,
        n_agents: int = 2,
        horizon: int = 16,
        seed: int | None = None,
        state_numbers: Sequence[int] | None = None,
        reward_win: float = 10.0,
    ) -> None:
        limits = tuple(state_numbers) if state_numbers is not None else (6,) * n_agents
        if len(limits) != n_agents or any(limit < 1 for limit in limits):
            raise ValueError("state_numbers must contain one positive limit per agent")
        self.n_agents = n_agents
        self.state_numbers = np.asarray(limits, dtype=np.int64)
        # The release uses max(state_numbers) + 10. Retain that paper-task horizon.
        self.horizon = max(int(horizon), int(self.state_numbers.max()) + 10)
        self.obs_dim = 1
        self.num_actions = 3
        self.reward_win = float(reward_win)
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self.positions = np.zeros(n_agents, dtype=np.int64)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self.positions = np.asarray(
            [self._rng.integers(1, limit + 1) for limit in self.state_numbers],
            dtype=np.int64,
        )
        return self._observations(), self._info(success=False)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action_array.shape}")
        if np.any((action_array < 0) | (action_array >= self.num_actions)):
            raise ValueError("actions must be in {0, 1, 2}")

        self._steps += 1
        self.positions = np.where(action_array == 1, self.positions - 1, self.positions)
        self.positions = np.where(action_array == 2, self.positions + 1, self.positions)
        self.positions = np.clip(self.positions, 0, self.state_numbers)

        all_arrived = bool(np.all(self.positions == 0))
        partial_arrival = bool(np.any(self.positions == 0)) and not all_arrived
        # Join1Env reports its episode limit through ``terminated`` rather than a
        # Gymnasium truncation, so its Q-learning target does not bootstrap here.
        terminated = all_arrived or partial_arrival or self._steps >= self.horizon
        truncated = False
        reward = self.reward_win if all_arrived else 0.0
        return self._observations(), reward, terminated, truncated, self._info(success=all_arrived)

    def close(self) -> None:
        return None

    def _observations(self) -> np.ndarray:
        return self.positions.astype(np.float32).reshape(self.n_agents, 1)

    def _info(self, *, success: bool) -> dict:
        normalized = self.positions / self.state_numbers
        return {
            "success": success,
            "mean_distance": float(normalized.mean()),
            "positions": self.positions.tolist(),
        }


__all__ = ["NDQHallwayEnv"]
