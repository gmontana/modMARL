"""Shared environment adapter for MAT-family on-policy sequence learners.

The algorithm packages own networks, rollout data, GAE, objectives, and
optimizers.  This module owns only interaction with the common environment
contract, deterministic held-out evaluation, checkpoint serialization, and the
bounded learning-evidence summary used by the curve tooling.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.mat import MATRollout


def train_transformer(
    *,
    algorithm: str,
    learner_factory: Callable[[], torch.nn.Module],
    env: str,
    n_agents: int,
    horizon: int,
    episodes: int,
    seed: int,
    rollout_episodes: int,
    ppo_epochs: int,
    num_minibatches: int,
    evaluation_episodes: int,
    device: str,
    checkpoint: str | None,
    algorithm_config: dict[str, object],
) -> dict:
    """Train one MAT-family learner and return reproducible curve evidence."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    learner = learner_factory().to(dev)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        learner,
        env,
        n_agents,
        horizon,
        evaluation_seed,
        evaluation_episodes,
        dev,
        random_policy=True,
    )

    returns: list[float] = []
    training_metrics: list[dict[str, float]] = []
    episode_count = 0
    update_count = 0
    total_updates = math.ceil(episodes / rollout_episodes)
    while episode_count < episodes:
        rollout = MATRollout(learner)
        count = min(rollout_episodes, episodes - episode_count)
        for _ in range(count):
            obs, _ = environment.reset(seed=seed + episode_count)
            episode_return = 0.0
            terminated = False
            for _ in range(environment.horizon):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                actions, log_probs, values = learner.act(obs_tensor)
                next_obs, reward, terminated, truncated, _ = environment.step(
                    actions.cpu().numpy(),
                )
                rollout.add(
                    obs=obs_tensor,
                    actions=actions,
                    log_probs=log_probs,
                    values=values,
                    team_reward=reward,
                )
                obs = next_obs
                episode_return += reward
                if terminated or truncated:
                    break
            final_mask = torch.zeros(environment.n_agents, device=dev) if terminated else torch.ones(
                environment.n_agents, device=dev,
            )
            if terminated:
                bootstrap = torch.zeros(environment.n_agents, device=dev)
            else:
                bootstrap = learner.values(torch.as_tensor(obs, dtype=torch.float32, device=dev))
            rollout.finish_episode(bootstrap, final_mask)
            returns.append(float(episode_return))
            episode_count += 1
        metrics = learner.update(
            rollout.batch(),
            epochs=ppo_epochs,
            num_minibatches=num_minibatches,
            training_step=update_count,
            total_steps=total_updates,
        )
        training_metrics.append(metrics)
        update_count += 1

    if checkpoint is not None:
        payload = {
            "algorithm": algorithm,
            "model": learner.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "episodes": episode_count,
            "updates": update_count,
        }
        if hasattr(learner, "edge_optimizer"):
            payload["edge_optimizer"] = learner.edge_optimizer.state_dict()
        torch.save(payload, checkpoint)

    final_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    result = {
        "algorithm": algorithm,
        "env": env,
        "episodes": episode_count,
        "n_agents": environment.n_agents,
        "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "training_metrics": training_metrics,
        "config": {
            "env": env,
            "n_agents": n_agents,
            "horizon": horizon,
            "episodes": episodes,
            "seed": seed,
            "rollout_episodes": rollout_episodes,
            "ppo_epochs": ppo_epochs,
            "num_minibatches": num_minibatches,
            "evaluation_episodes": evaluation_episodes,
            "device": device,
            **algorithm_config,
        },
        "validation_criterion": {
            "return_improvement_fraction_of_abs_random": 0.2,
            "scope": "every confirmation seed",
        },
    }
    if hasattr(learner, "communication_rate"):
        result["communication_rate"] = learner.communication_rate()
    return result


def _evaluate(
    learner,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    random_policy: bool = False,
) -> dict[str, list[float]]:
    evaluation_env = make_env(env_name, n_agents, horizon, seed)
    returns, successes, distances = [], [], []
    random_generator = np.random.default_rng(seed)
    learner.eval()
    for episode in range(episodes):
        obs, _ = evaluation_env.reset(seed=seed + episode)
        episode_return, info = 0.0, {}
        for _ in range(evaluation_env.horizon):
            if random_policy:
                actions = random_generator.integers(
                    evaluation_env.num_actions, size=evaluation_env.n_agents,
                )
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions, _, _ = learner.act(obs_tensor, deterministic=True)
                actions = actions.cpu().numpy()
            obs, reward, terminated, truncated, info = evaluation_env.step(actions)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    learner.train()
    return {"returns": returns, "successes": successes, "mean_distances": distances}
