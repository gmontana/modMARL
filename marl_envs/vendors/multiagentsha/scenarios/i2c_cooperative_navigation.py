"""I2C's seven-agent Cooperative Navigation scenario.

This is the authors' ``multiagent/scenarios/cn.py`` scenario adapted only to the
vendored module namespace. Each agent observes itself, its three nearest teammates,
and its three nearest landmarks. The reward, initialization, and 7-agent/7-landmark
task definition are unchanged.
"""

import numpy as np

from multiagentsha.core import Agent, Landmark, World
from multiagentsha.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        world.range_p = 1
        world.num_agents_obs = 3
        world.num_landmarks_obs = 3
        world.collaborative = False
        world.discrete_action = True
        world.agents = [Agent() for _ in range(7)]
        for index, agent in enumerate(world.agents):
            agent.name = f"agent {index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        world.landmarks = [Landmark() for _ in range(7)]
        for index, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark {index}"
            landmark.collide = False
            landmark.movable = False
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np.random.uniform(-world.range_p, world.range_p, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for index, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = np.random.uniform(-world.range_p, world.range_p, world.dim_p)
            if index:
                while any(
                    np.linalg.norm(landmark.state.p_pos - previous.state.p_pos) <= 0.22
                    for previous in world.landmarks[:index]
                ):
                    landmark.state.p_pos = np.random.uniform(-world.range_p, world.range_p, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        min_dists = sum(
            min(np.linalg.norm(other.state.p_pos - landmark.state.p_pos) for other in world.agents)
            for landmark in world.landmarks
        )
        occupied = sum(
            min(np.linalg.norm(other.state.p_pos - landmark.state.p_pos) for other in world.agents)
            < agent.size + landmark.size
            for landmark in world.landmarks
        )
        collisions = 0.0
        if agent.collide:
            collisions = 0.5 * sum(
                self.is_collision(first, second)
                for first in world.agents for second in world.agents if first is not second
            )
        return -min_dists, collisions, min_dists, occupied

    def is_collision(self, first, second):
        return np.linalg.norm(first.state.p_pos - second.state.p_pos) < first.size + second.size

    def reward(self, agent, world):
        reward = -sum(
            min(np.linalg.norm(other.state.p_pos - landmark.state.p_pos) for other in world.agents)
            for landmark in world.landmarks
        )
        if agent.collide:
            reward -= 0.5 * sum(
                self.is_collision(first, second)
                for first in world.agents for second in world.agents if first is not second
            )
        return reward

    def observation(self, agent, world):
        nearest_landmarks = sorted(
            (landmark.state.p_pos - agent.state.p_pos for landmark in world.landmarks),
            key=np.linalg.norm,
        )[: world.num_landmarks_obs]
        nearest_agents = sorted(
            (other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent),
            key=np.linalg.norm,
        )[: world.num_agents_obs]
        return np.concatenate([agent.state.p_vel, agent.state.p_pos, *nearest_landmarks, *nearest_agents])
