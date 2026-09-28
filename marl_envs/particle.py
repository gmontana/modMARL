from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

VENDOR_ROOT = Path(__file__).resolve().parent / "vendors"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


PAPER_PARTICLE_ENVS: dict[str, dict[str, Any]] = {
    "paper_i2c_navigation": {
        "scenario": "i2c_cooperative_navigation",
        "horizon": 40,
    },
    "paper_maddpg_navigation": {
        "scenario": "simple_spread",
        "horizon": 25,
        "shared_reward": True,
    },
    "paper_navigation": {
        "scenario": "simple_spread",
        "horizon": 25,
    },
    "paper_mdmaddpg_navigation": {
        "scenario": "simple_spread",
        "horizon": 100,
        "n_agents": 2,
    },
    "paper_formation": {
        "scenario": "simple_formation_po",
        "horizon": 50,
    },
    "paper_line": {
        "scenario": "simple_line_po",
        "horizon": 50,
    },
    "paper_pack": {
        "scenario": "simple_spread_pack_leader_4_2",
        "horizon": 50,
    },
}


def paper_particle_env_available() -> bool:
    try:
        importlib.import_module("multiagentsha.environment")
        _load_scenario_module("simple_spread")
    except ImportError:
        return False
    return True


def _load_scenario_module(name: str):
    scenario_path = VENDOR_ROOT / "multiagentsha" / "scenarios" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"multiagentsha.scenarios.{name}", scenario_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load scenario: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_multiagentsha_env(scenario_name: str, n_agents: int | None = None):
    from multiagentsha.environment import MultiAgentEnv

    scenario_module = _load_scenario_module(scenario_name)
    scenario = scenario_module.Scenario()
    world = scenario.make_world()
    if n_agents is not None:
        if not 1 <= n_agents <= len(world.agents):
            raise ValueError(
                f"scenario {scenario_name!r} cannot provide {n_agents} agents",
            )
        world.agents = world.agents[:n_agents]
        world.landmarks = world.landmarks[:n_agents]
        scenario.reset_world(world)
    return MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        scenario.benchmark_data,
        discrete_action=True,
    )


class PaperParticleEnv:
    """Adapter from the archived CDC particle env fork to the local trainer contract."""

    def __init__(self, env_name: str, *, horizon: int | None = None, seed: int | None = None) -> None:
        if env_name not in PAPER_PARTICLE_ENVS:
            raise ValueError(f"unknown paper env: {env_name}")
        config = PAPER_PARTICLE_ENVS[env_name]
        self.env_name = env_name
        self.scenario_name = config["scenario"]
        del horizon
        self.horizon = int(config["horizon"])
        self._env = _make_multiagentsha_env(
            self.scenario_name, config.get("n_agents"),
        )
        self._env.shared_reward = bool(config.get("shared_reward", False))
        self.shared_reward = self._env.shared_reward
        if hasattr(self._env, "_seed"):
            self._env._seed(seed)
        self.n_agents = int(self._env.n)
        self.obs_dim = int(self._env.observation_space[0].shape[0])
        self.num_actions = int(self._env.action_space[0].n)
        self.step_count = 0
        self.targets_caught = 0
        self.solve_step: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None and hasattr(self._env, "_seed"):
            self._env._seed(seed)
        self.step_count = 0
        self.targets_caught = 0
        self.solve_step = None
        if hasattr(self._env, "_reset"):
            obs_n = self._env._reset()
        else:
            obs_n = self._env.reset()
        obs = np.asarray(obs_n, dtype=np.float32)
        return obs, self._empty_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action)
        if action.shape == (self.n_agents,):
            action_vectors = np.eye(self.num_actions, dtype=np.float32)[action.astype(np.int64)]
        elif action.shape == (self.n_agents, self.num_actions):
            # OpenAI MADDPG executes its soft Gumbel samples directly in MPE.
            action_vectors = action.astype(np.float32)
        else:
            expected = f"{(self.n_agents,)} or {(self.n_agents, self.num_actions)}"
            raise ValueError(f"expected action shape {expected}, got {action.shape}")
        if hasattr(self._env, "_step"):
            obs_n, reward_n, _, info_n = self._env._step(list(action_vectors))
        else:
            obs_n, reward_n, _, info_n = self._env.step(list(action_vectors))
        self.step_count += 1
        obs = np.asarray(obs_n, dtype=np.float32)
        reward_values = np.asarray(reward_n, dtype=np.float32)
        reward = float(reward_values[0] if self.shared_reward else reward_values.mean())
        info = self._parse_info(info_n)
        info["agent_rewards"] = reward_values.copy()
        terminated = False
        truncated = self.step_count >= self.horizon
        return obs, reward, terminated, truncated, info

    def communication_candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return receiver-relative teammate locations and eligible sender mask.

        I2C's Cooperative Navigation scenario exposes only the three nearest
        teammates to each receiver. Other paper particle tasks expose every peer.
        """
        positions = np.stack([agent.state.p_pos for agent in self._env.world.agents])
        # Authors' get_comm_pairs uses receiver_position - sender_position.
        relative = positions[:, None, :] - positions[None, :, :]
        eligible = ~np.eye(self.n_agents, dtype=bool)
        if self.env_name == "paper_i2c_navigation":
            nearest = np.argsort(np.linalg.norm(relative, axis=-1), axis=1)[:, 1:4]
            eligible[:] = False
            eligible[np.arange(self.n_agents)[:, None], nearest] = True
        return relative.astype(np.float32), eligible

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def _empty_info(self) -> dict[str, Any]:
        return {
            "success": None if self.env_name == "paper_pack" else False,
            "mean_distance": None,
            "collisions": 0.0,
            "targets_caught": 0.0,
            "time_to_solve": None,
        }

    def _parse_info(self, info_n: dict[str, list[Any]]) -> dict[str, Any]:
        payload = info_n["n"][0] if isinstance(info_n, dict) and info_n.get("n") else ()
        if self.env_name == "paper_i2c_navigation":
            reward, collisions, sum_min_dists, occupied_landmarks = payload
            mean_distance = float(sum_min_dists) / max(1, self.n_agents)
            success = bool(occupied_landmarks == self.n_agents)
            if success and self.solve_step is None:
                self.solve_step = self.step_count
            return {
                "success": success,
                "mean_distance": mean_distance,
                "collisions": float(collisions),
                "targets_caught": 0.0,
                "time_to_solve": self.solve_step,
                "benchmark_reward": float(reward),
            }
        if self.env_name in {
            "paper_navigation", "paper_maddpg_navigation", "paper_mdmaddpg_navigation",
        }:
            reward, collisions, sum_min_dists, occupied_landmarks, _ = payload
            mean_distance = float(sum_min_dists) / max(1, self.n_agents)
            success = bool(occupied_landmarks == self.n_agents)
            if success and self.solve_step is None:
                self.solve_step = self.step_count
            return {
                "success": success,
                "mean_distance": mean_distance,
                "collisions": float(collisions),
                "targets_caught": 0.0,
                "time_to_solve": self.solve_step,
                "benchmark_reward": float(reward),
            }
        if self.env_name in {"paper_formation", "paper_line"}:
            reward, mean_distance, success, _ = payload
            success = bool(success)
            if success and self.solve_step is None:
                self.solve_step = self.step_count
            return {
                "success": success,
                "mean_distance": float(mean_distance),
                "collisions": 0.0,
                "targets_caught": 0.0,
                "time_to_solve": self.solve_step,
                "benchmark_reward": float(reward),
            }
        reward, collisions, _, max_distance, occupied_landmarks, _ = payload
        self.targets_caught += int(occupied_landmarks)
        return {
            "success": None,
            "mean_distance": float(max_distance),
            "collisions": float(collisions),
            "targets_caught": float(self.targets_caught),
            "time_to_solve": None,
            "benchmark_reward": float(reward),
        }
