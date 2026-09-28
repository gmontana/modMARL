"""MADDPG-M's fixed-broadcasting noisy Cooperative Navigation task.

Design: N agents must cover N landmarks. Only one **gifted** agent observes the true
landmark positions; every other agent observes a noisy version. No agent knows whether it is
gifted. The extrinsic reward is the standard cooperative-navigation reward on the *true*
landmarks. An **intrinsic reward** — the (negative) distance from the agents to the landmarks
that a chosen agent *believes* in — is exposed for algorithms like MADDPG-M that share an
observation through a communication medium: only by broadcasting the gifted agent's
observation can the team be intrinsically rewarded for reaching the true landmarks.

Model: continuous 2-D agents cover the same number of landmarks. Every observation contains
velocity, position, relative believed landmarks, and relative teammate positions; only the gifted
agent's landmark beliefs are correct. ``NoisyNavigationEnv`` uses four [0, 1] direction magnitudes;
``SignedNoisyNavigationEnv`` exposes the same dynamics through a unique signed 2-D action.
Invariants: extrinsic reward uses true landmarks; intrinsic reward uses the broadcaster's beliefs.
Why: this is the paper's simplest fixed-broadcasting scenario, where communication is essential.
"""

from __future__ import annotations

import numpy as np


class NoisyNavigationEnv:
    def __init__(
        self,
        n_agents: int = 2,
        horizon: int = 25,
        arena_size: float = 1.0,
        noise: float = 1.0,
        gifted_agent: int = 0,
        step_size: float = 0.1,
        success_radius: float = 0.15,
        seed: int | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.n_landmarks = n_agents
        self.horizon = horizon
        self.arena_size = arena_size
        self.noise = noise
        self.gifted_agent = gifted_agent
        self.step_size = step_size
        self.success_radius = success_radius
        self.num_actions = 4
        self.obs_dim = 4 + 2 * self.n_landmarks + 2 * (self.n_agents - 1)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._positions = np.zeros((n_agents, 2), dtype=np.float32)
        self._velocities = np.zeros((n_agents, 2), dtype=np.float32)
        self._landmarks = np.zeros((self.n_landmarks, 2), dtype=np.float32)
        self._believed = np.zeros((n_agents, self.n_landmarks, 2), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._positions = self._rng.uniform(-self.arena_size, self.arena_size, size=(self.n_agents, 2)).astype(np.float32)
        self._velocities.fill(0.0)
        self._landmarks = self._rng.uniform(-self.arena_size, self.arena_size, size=(self.n_landmarks, 2)).astype(np.float32)

        # Each agent's *believed* landmarks: true for the gifted agent, noisy for the rest.
        self._believed = np.repeat(self._landmarks[None], self.n_agents, axis=0).astype(np.float32)
        for i in range(self.n_agents):
            if i != self.gifted_agent:
                self._believed[i] += self.noise * self._rng.standard_normal(self._believed[i].shape).astype(np.float32)

        return self._obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.n_agents, 4)
        action = np.clip(action, 0.0, 1.0)
        self._velocities = np.stack(
            [action[:, 1] - action[:, 0], action[:, 3] - action[:, 2]], axis=-1,
        )
        self._positions = np.clip(
            self._positions + self.step_size * self._velocities, -self.arena_size, self.arena_size,
        ).astype(np.float32)
        self._t += 1
        reward = self._extrinsic_reward()
        truncated = self._t >= self.horizon
        return self._obs(), reward, False, truncated, self._info()

    def intrinsic_reward(self, agent_index: int) -> float:
        """Negative mean distance from the agents to the landmarks ``agent_index`` believes in.

        MADDPG-M broadcasts one agent's observation into the medium; this rewards the team for
        reaching *those* (possibly noisy) landmarks, regardless of whether they are the true ones.
        """
        return float(-self._min_distances(self._believed[agent_index]).sum() - self._collision_penalty())

    def close(self) -> None:
        pass

    def _obs(self) -> np.ndarray:
        relative_landmarks = self._believed - self._positions[:, None, :]
        relative_agents = self._positions[None, :, :] - self._positions[:, None, :]
        peer_mask = ~np.eye(self.n_agents, dtype=bool)
        peers = relative_agents[peer_mask].reshape(self.n_agents, self.n_agents - 1, 2)
        return np.concatenate(
            [self._velocities, self._positions, relative_landmarks.reshape(self.n_agents, -1), peers.reshape(self.n_agents, -1)],
            axis=1,
        ).astype(np.float32)

    def _min_distances(self, landmarks: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(landmarks[:, None, :] - self._positions[None, :, :], axis=-1)
        return dists.min(axis=1)                                          # nearest agent per landmark

    def _extrinsic_reward(self) -> float:
        return float(-self._min_distances(self._landmarks).sum() - self._collision_penalty())

    def _collision_penalty(self) -> float:
        distances = np.linalg.norm(self._positions[:, None] - self._positions[None, :], axis=-1)
        collisions = (distances < 0.1) & ~np.eye(self.n_agents, dtype=bool)
        return float(collisions.sum() / 2)

    def _info(self) -> dict:
        min_dists = self._min_distances(self._landmarks)
        return {"mean_distance": float(np.mean(min_dists)), "success": bool(np.all(min_dists < self.success_radius))}


class SignedNoisyNavigationEnv(NoisyNavigationEnv):
    """Noisy navigation with non-redundant signed ``(x, y)`` continuous actions."""

    def __init__(
        self,
        n_agents: int = 2,
        horizon: int = 25,
        arena_size: float = 1.0,
        noise: float = 1.0,
        gifted_agent: int = 0,
        step_size: float = 0.1,
        success_radius: float = 0.15,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_agents=n_agents,
            horizon=horizon,
            arena_size=arena_size,
            noise=noise,
            gifted_agent=gifted_agent,
            step_size=step_size,
            success_radius=success_radius,
            seed=seed,
        )
        self.num_actions = 2
        self.action_low = -1.0
        self.action_high = 1.0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance from ``action: (n_agents, 2)`` in the closed interval ``[-1, 1]``."""
        signed = np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.n_agents, 2),
            self.action_low,
            self.action_high,
        )
        paired = np.stack(
            [
                np.maximum(-signed[:, 0], 0.0),
                np.maximum(signed[:, 0], 0.0),
                np.maximum(-signed[:, 1], 0.0),
                np.maximum(signed[:, 1], 0.0),
            ],
            axis=-1,
        )
        return super().step(paired)


__all__ = ["NoisyNavigationEnv", "SignedNoisyNavigationEnv"]
