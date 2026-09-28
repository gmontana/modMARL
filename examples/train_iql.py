"""Train the paper-faithful deep IQL implementation on a modMARL environment.

This file only owns environment interaction and each learner's private replay
memory. Networks, exploration, TD loss, optimizer, and target synchronization are
all defined in ``modmarl.algorithms.iql``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.iql import IQLAgent
from modmarl.common.replay import ReplayBuffer


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    learning_rate: float = 2.5e-4,
    hidden_dim: int = 128,
    buffer_size: int = 1_000_000,
    batch_size: int = 32,
    learn_start: int = 50_000,
    update_every: int = 4,
    target_update_interval: int = 10_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_anneal_steps: int = 1_000_000,
    gamma: float = 0.99,
    evaluation_episodes: int = 32,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    n = environment.n_agents
    learner = IQLAgent(
        n, environment.obs_dim, environment.num_actions, hidden_dim,
        learning_rate=learning_rate, gamma=gamma,
        target_update_interval=target_update_interval,
        epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        epsilon_anneal_steps=epsilon_anneal_steps, learn_start=learn_start,
    ).to(dev)
    replays = [ReplayBuffer(buffer_size, 1, environment.obs_dim) for _ in range(n)]
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        learner, env, n, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        learner, env, n, horizon, evaluation_seed, evaluation_episodes, dev, random_policy=True,
    )

    total_steps = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        for _ in range(environment.horizon):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            actions = learner.act(obs_tensor, learner.epsilon(total_steps)).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = environment.step(actions)
            for index, replay in enumerate(replays):
                replay.add(
                    obs=obs[index:index + 1], actions=actions[index:index + 1], reward=reward,
                    next_obs=next_obs[index:index + 1], done=terminated,
                )
            obs = next_obs
            episode_return += reward
            total_steps += 1

            if total_steps > learn_start and total_steps % update_every == 0:
                for index, replay in enumerate(replays):
                    if len(replay) >= batch_size:
                        learner.update_agent(index, replay.sample(batch_size, dev))
            learner.maybe_update_targets(total_steps)
            if terminated or truncated:
                break
        returns.append(episode_return)

    if checkpoint is not None:
        torch.save(
            {
                "model": learner.state_dict(),
                "optimizers": [optimizer.state_dict() for optimizer in learner.optimizers],
                "episodes": episodes,
                "total_steps": total_steps,
            },
            checkpoint,
        )
    final_evaluation = _evaluate(
        learner, env, n, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    return {
        "algorithm": "iql", "env": env, "episodes": episodes, "n_agents": n,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns, "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
    }


def _evaluate(
    learner: IQLAgent,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    random_policy: bool = False,
) -> dict[str, list[float]]:
    """Run exploration-free held-out episodes and retain standard task metrics."""
    returns: list[float] = []
    successes: list[float] = []
    mean_distances: list[float] = []
    evaluation_env = make_env(env_name, n_agents, horizon, seed)
    cuda_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        random_generator = np.random.default_rng(seed)
        for episode in range(episodes):
            obs, _ = evaluation_env.reset(seed=seed + episode)
            episode_return = 0.0
            info: dict = {}
            for _ in range(evaluation_env.horizon):
                if random_policy:
                    actions = random_generator.integers(
                        evaluation_env.num_actions, size=evaluation_env.n_agents,
                    )
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    actions = learner.act(obs_tensor, epsilon=0.0).cpu().numpy()
                obs, reward, terminated, truncated, info = evaluation_env.step(actions)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            successes.append(float(bool(info.get("success", False))))
            mean_distances.append(float(info.get("mean_distance", float("nan"))))
    return {"returns": returns, "successes": successes, "mean_distances": mean_distances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train independent deep Q-learners.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="iql.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
