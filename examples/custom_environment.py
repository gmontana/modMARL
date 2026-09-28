"""Runnable environment-adapter tutorial: python examples/custom_environment.py.

This is a 16-episode interface check with one learning update, not learning evidence.
Replace PrivateBitSimulator with your simulator while keeping LocalObservationAdapter.
"""

from __future__ import annotations

import numpy as np
import torch

from modmarl import TarMACAgent, TarMACConfig


class PrivateBitSimulator:
    """Example external simulator with named agents and dictionary observations."""

    agents = ("leader", "follower")

    def reset(self, seed):
        self.bit = int(np.random.default_rng(seed).integers(2))
        return {"leader": [1.0, float(self.bit)], "follower": [0.0, 0.0]}

    def step(self, actions):
        success = all(actions[name] == self.bit for name in self.agents)
        observations = {"leader": [1.0, float(self.bit)], "follower": [0.0, 0.0]}
        return observations, float(success), True, False, {"success": success}


class LocalObservationAdapter:
    """Stable agent order, local observations, discrete actions and one team reward."""

    n_agents, obs_dim, num_actions, horizon = 2, 2, 2, 1

    def __init__(self):
        self.simulator = PrivateBitSimulator()

    def _stack(self, observations):
        return np.asarray([observations[name] for name in self.simulator.agents], dtype=np.float32)

    def reset(self, *, seed=None):
        return self._stack(self.simulator.reset(seed)), {}

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.n_agents,) or np.any((actions < 0) | (actions >= self.num_actions)):
            raise ValueError("Expected one valid discrete action per agent")
        named = {name: int(actions[index]) for index, name in enumerate(self.simulator.agents)}
        obs, reward, terminated, truncated, info = self.simulator.step(named)
        return self._stack(obs), reward, terminated, truncated, info

    def close(self):
        pass


def run():
    torch.manual_seed(0)
    env = LocalObservationAdapter()
    agent = TarMACAgent(env.n_agents, env.obs_dim, env.num_actions,
                       TarMACConfig(communication_rounds=2))
    observations, actions, rewards = [], [], []
    for seed in range(16):
        obs, _ = env.reset(seed=seed)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        state = agent.policy.initial_state(1, env.n_agents, torch.device("cpu"))
        with torch.no_grad():
            action, _, _, _, _ = agent.policy.act(obs_tensor, state)
        _, reward, terminated, truncated, _ = env.step(action[0].numpy())
        assert terminated and not truncated
        observations.append(obs)
        actions.append(action[0])
        rewards.append(reward)
    metrics = agent.update(
        obs=torch.from_numpy(np.stack(observations)).unsqueeze(1),
        actions=torch.stack(actions).unsqueeze(1),
        rewards=torch.tensor(rewards).unsqueeze(1),
        mask=torch.ones(16, 1), continuation=torch.zeros(16, 1),
    )
    env.close()
    print("Environment contract and one update completed:", metrics)
    return metrics


if __name__ == "__main__":
    torch.set_num_threads(1)
    run()
