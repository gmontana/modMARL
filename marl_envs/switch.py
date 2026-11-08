"""The switch riddle from Foerster et al. (NeurIPS 2016), Section 6.2.

Model: each day the warden sends one prisoner, chosen uniformly at random with
replacement, to the interrogation room. Only that prisoner acts: "None" or
"Tell". The light switch is a one-bit channel from the previous day's prisoner
to the current one; nobody else hears anything.
Invariants: "Tell" ends the episode with +1 when every prisoner has visited
the room (counting the current day) and -1 otherwise. Reaching the final day
without "Tell" ends the episode with reward 0. The horizon is T = 4n - 6.
Interface: the standard local ``reset``/``step`` API plus three per-step
queries used by communication learners: ``available_actions`` (non-visitors
may only choose "None"), ``communication_mask`` (receiver x sender) and
``message_senders``.
Why: the riddle is DIAL and RIAL's canonical task and the smallest problem in
which agents must invent a protocol. This port follows the authors' release
(``iassael/learning-to-communicate`` at ``1cdfc235``, ``code/game/Switch.lua``):
visitors are drawn up front for all T days, observations distinguish "in the
room" from "elsewhere", and the visitor on day t > 1 hears the visitor of
day t - 1 even when the same prisoner was sent twice.
"""

from __future__ import annotations

import numpy as np

NONE, TELL = 0, 1


class SwitchRiddleEnv:
    """Batched-free single-episode switch riddle with the release's channel rule."""

    def __init__(self, n_agents: int = 3, horizon: int | None = None, seed: int | None = None) -> None:
        if n_agents < 2:
            raise ValueError("the switch riddle needs at least two prisoners")
        self.n_agents = n_agents
        self.horizon = int(horizon) if horizon is not None else 4 * n_agents - 6
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        self.obs_dim = 2          # one-hot: [in the interrogation room, elsewhere]
        self.num_actions = 2      # NONE, TELL
        self._rng = np.random.default_rng(seed)
        self.visitors = np.zeros(self.horizon, dtype=np.int64)
        self.day = 0
        self.done = False

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.visitors = self._rng.integers(0, self.n_agents, size=self.horizon)
        self.day = 0
        self.done = False
        return self._observations(), self._info(success=False)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("step() called after the episode ended; call reset()")
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.n_agents,):
            raise ValueError(f"expected action shape {(self.n_agents,)}, got {action_array.shape}")
        available = self.available_actions()
        chosen = available[np.arange(self.n_agents), np.clip(action_array, 0, self.num_actions - 1)]
        if np.any((action_array < 0) | (action_array >= self.num_actions)) or not np.all(chosen):
            raise ValueError("only today's visitor may choose Tell; everyone else must choose None")

        visitor = int(self.visitors[self.day])
        reward, success = 0.0, False
        if action_array[visitor] == TELL:
            success = self.everyone_has_visited(self.day)
            reward = 1.0 if success else -1.0
            self.done = True
        elif self.day == self.horizon - 1:
            self.done = True
        self.day += 1
        observations = self._observations() if not self.done else np.zeros(
            (self.n_agents, self.obs_dim), dtype=np.float32,
        )
        return observations, reward, self.done, False, self._info(success=success)

    def everyone_has_visited(self, day: int | None = None) -> bool:
        """Whether every prisoner has been in the room on or before ``day``."""
        last = self.horizon - 1 if day is None else day
        return len(np.unique(self.visitors[: last + 1])) == self.n_agents

    def available_actions(self) -> np.ndarray:
        available = np.zeros((self.n_agents, self.num_actions), dtype=np.float32)
        available[:, NONE] = 1.0
        if not self.done:
            available[self.visitors[min(self.day, self.horizon - 1)], TELL] = 1.0
        return available

    def communication_mask(self) -> np.ndarray:
        """Return ``mask[receiver, sender]`` for messages delivered on the current day."""
        mask = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        if 0 < self.day < self.horizon and not self.done:
            mask[self.visitors[self.day], self.visitors[self.day - 1]] = 1.0
        return mask

    def message_senders(self) -> np.ndarray:
        """Return which agents can set the switch today: only the current visitor."""
        senders = np.zeros(self.n_agents, dtype=np.float32)
        if not self.done:
            senders[self.visitors[min(self.day, self.horizon - 1)]] = 1.0
        return senders

    def state(self) -> np.ndarray:
        """Centralized state: today's visitor (one-hot), visit history and progress."""
        visitor = np.zeros(self.n_agents, dtype=np.float32)
        visited = np.zeros(self.n_agents, dtype=np.float32)
        day = min(self.day, self.horizon - 1)
        visitor[self.visitors[day]] = 1.0
        visited[np.unique(self.visitors[: day + 1])] = 1.0
        return np.concatenate([visitor, visited, [day / max(self.horizon - 1, 1)]]).astype(np.float32)

    def close(self) -> None:
        return None

    def _observations(self) -> np.ndarray:
        observations = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        observations[:, 1] = 1.0
        visitor = self.visitors[self.day]
        observations[visitor] = (1.0, 0.0)
        return observations

    def _info(self, *, success: bool) -> dict:
        return {
            "success": bool(success),
            "everyone_visited": self.everyone_has_visited(),
            "day": int(self.day),
        }


__all__ = ["NONE", "TELL", "SwitchRiddleEnv"]
