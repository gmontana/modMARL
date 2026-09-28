"""Train the paper-faithful recurrent VDN implementation.

This runner owns only environment interaction and generic whole-episode replay.
VDN's network, recurrence, exploration, lambda return, optimizer, and target
updates are defined in ``modmarl.algorithms.vdn``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.vdn import VDNAgent
from modmarl.common.replay import EpisodeReplayBuffer


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    learning_rate: float = 1e-4,
    hidden_dim: int = 32,
    buffer_size: int = 5000,
    batch_size: int = 32,
    warmup_episodes: int = 32,
    updates_per_episode: int = 1,
    target_update_interval: int = 200,
    epsilon_anneal_steps: int = 50_000,
    gamma: float = 0.99,
    trace_lambda: float = 0.9,
    trace_length: int = 8,
    evaluation_episodes: int = 32,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    learner = VDNAgent(
        environment.n_agents, environment.obs_dim, environment.num_actions, hidden_dim,
        learning_rate=learning_rate, gamma=gamma, trace_lambda=trace_lambda,
        trace_length=trace_length, target_update_interval=target_update_interval,
        epsilon_anneal_steps=epsilon_anneal_steps,
    ).to(dev)
    replay = EpisodeReplayBuffer(
        buffer_size, horizon, environment.n_agents, environment.obs_dim,
    )
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        learner, env, environment.n_agents, horizon, evaluation_seed,
        evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        learner, env, environment.n_agents, horizon, evaluation_seed,
        evaluation_episodes, dev, random_policy=True,
    )
    total_steps = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        learner.reset_hidden()
        observations = [obs.copy()]
        actions, rewards, dones = [], [], []
        episode_return = 0.0
        for _ in range(environment.horizon):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            action = learner.act(obs_tensor, learner.epsilon(total_steps)).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = environment.step(action)
            observations.append(next_obs.copy())
            actions.append(action.copy())
            rewards.append(reward)
            dones.append(float(terminated))
            obs = next_obs
            episode_return += reward
            total_steps += 1
            if terminated or truncated:
                break
        replay.add_episode(
            obs=np.asarray(observations), actions=np.asarray(actions),
            rewards=np.asarray(rewards), dones=np.asarray(dones),
        )
        if len(replay) >= max(batch_size, warmup_episodes):
            for _ in range(updates_per_episode):
                learner.update(replay.sample(batch_size, dev))
        learner.maybe_update_targets(episode + 1)
        returns.append(episode_return)

    if checkpoint is not None:
        torch.save(
            {"model": learner.state_dict(), "optimizer": learner.optimizer.state_dict(),
             "episodes": episodes, "total_steps": total_steps},
            checkpoint,
        )
    final_evaluation = _evaluate(
        learner, env, environment.n_agents, horizon, evaluation_seed,
        evaluation_episodes, dev,
    )
    return {
        "algorithm": "vdn", "env": env, "episodes": episodes,
        "n_agents": environment.n_agents,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns, "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
    }


def _evaluate(
    learner: VDNAgent,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    random_policy: bool = False,
) -> dict[str, list[float]]:
    """Evaluate recurrent decentralized policies on fixed held-out episodes."""
    evaluation_env = make_env(env_name, n_agents, horizon, seed)
    returns, successes, mean_distances = [], [], []
    cuda_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        random_generator = np.random.default_rng(seed)
        for episode in range(episodes):
            obs, _ = evaluation_env.reset(seed=seed + episode)
            learner.reset_hidden()
            episode_return = 0.0
            info: dict = {}
            for _ in range(evaluation_env.horizon):
                if random_policy:
                    action = random_generator.integers(
                        evaluation_env.num_actions, size=evaluation_env.n_agents,
                    )
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    action = learner.act(obs_tensor, epsilon=0.0).cpu().numpy()
                obs, reward, terminated, truncated, info = evaluation_env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            successes.append(float(bool(info.get("success", False))))
            mean_distances.append(float(info.get("mean_distance", float("nan"))))
    return {"returns": returns, "successes": successes, "mean_distances": mean_distances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recurrent VDN.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="vdn.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents,
          seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
