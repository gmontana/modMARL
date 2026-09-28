"""Train released recurrent CommNet with cooperative REINFORCE.

Run:
    python examples/train_commnet.py --episodes 300

Complete episodes form the paper's 288-game aggregate batch. Hidden, cell, and
previous-step communication states are reset to their released initial values at
every episode boundary.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl import CommNetAgent


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    learning_rate: float = 1e-3,
    hidden_dim: int = 50,
    batch_size: int = 288,
    unroll_length: int = 10,
    unroll_frequency: int = 4,
    evaluation_episodes: int = 32,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    agent = CommNetAgent(
        environment.obs_dim,
        environment.num_actions,
        hidden_dim,
        learning_rate=learning_rate,
        unroll_length=unroll_length,
        unroll_frequency=unroll_frequency,
    ).to(dev)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        agent,
        env,
        n_agents,
        horizon,
        evaluation_seed,
        evaluation_episodes,
        dev,
        random_policy=True,
    )

    returns: list[float] = []
    completed = 0
    while completed < episodes:
        rollout_size = min(batch_size, episodes - completed)
        episodes_batch = []
        for batch_index in range(rollout_size):
            episode, episode_return = _collect_episode(
                agent,
                environment,
                n_agents,
                seed + completed + batch_index,
                dev,
            )
            episodes_batch.append(episode)
            returns.append(episode_return)
        agent.update(**_pad_episodes(episodes_batch, n_agents, environment.obs_dim, dev))
        completed += rollout_size
        print(f"episodes {completed:4d}  return {returns[-1]:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {"agent": agent.state_dict(), "optimizer": agent.optimizer.state_dict()}, checkpoint,
        )
    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    return {
        "algorithm": "commnet",
        "env": env,
        "episodes": completed,
        "n_agents": n_agents,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "communication_rate": final_evaluation["communication_rate"],
        "validation_criterion": {
            "return_improvement_fraction_of_abs_random": 0.2,
            "scope": "every confirmation seed",
        },
        "config": {
            "model": "lstm",
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "unroll_length": unroll_length,
            "unroll_frequency": unroll_frequency,
            "initial_hidden": 0.1,
            "baseline_coefficient": agent.baseline_coefficient,
            "optimizer": "RMSprop",
            "rmsprop_alpha": 0.97,
            "rmsprop_epsilon": 1e-8,
        },
    }


@torch.no_grad()
def _collect_episode(agent, environment, n_agents, episode_seed, device):
    obs, _ = environment.reset(seed=episode_seed)
    hidden, cell, received = agent.actor.initial_state(1, n_agents, device)
    steps = {key: [] for key in ("obs", "actions", "rewards", "alive")}
    episode_return = 0.0
    alive = torch.ones(1, n_agents, device=device)
    for _ in range(environment.horizon):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action, hidden, cell, received = agent.act(
            obs_tensor.unsqueeze(0), hidden, cell, received, alive,
        )
        next_obs, reward, terminated, truncated, info = environment.step(
            action.squeeze(0).cpu().numpy(),
        )
        steps["obs"].append(obs_tensor)
        steps["actions"].append(action.squeeze(0))
        steps["rewards"].append(torch.tensor(float(reward), device=device))
        steps["alive"].append(alive.squeeze(0))
        obs = next_obs
        episode_return += float(reward)
        alive = torch.as_tensor(
            info.get("alive_mask", np.ones(n_agents)), dtype=torch.float32, device=device,
        ).unsqueeze(0)
        if terminated or truncated:
            break
    return {key: torch.stack(values) for key, values in steps.items()}, episode_return


def _pad_episodes(episodes, n_agents, obs_dim, device):
    batch_size = len(episodes)
    horizon = max(episode["obs"].shape[0] for episode in episodes)
    obs = torch.zeros(batch_size, horizon, n_agents, obs_dim, device=device)
    actions = torch.zeros(batch_size, horizon, n_agents, dtype=torch.long, device=device)
    rewards = torch.zeros(batch_size, horizon, device=device)
    mask = torch.zeros(batch_size, horizon, n_agents, device=device)
    for index, episode in enumerate(episodes):
        length = episode["obs"].shape[0]
        obs[index, :length] = episode["obs"]
        actions[index, :length] = episode["actions"]
        rewards[index, :length] = episode["rewards"]
        mask[index, :length] = episode["alive"]
    return {"obs": obs, "actions": actions, "rewards": rewards, "mask": mask}


@torch.no_grad()
def _evaluate(
    agent,
    env_name,
    n_agents,
    horizon,
    seed,
    episodes,
    device,
    *,
    random_policy: bool = False,
):
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    episode_returns, successes, distances = [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        hidden, cell, received = agent.actor.initial_state(1, n_agents, device)
        alive = torch.ones(1, n_agents, device=device)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                actions = generator.integers(environment.num_actions, size=n_agents)
            else:
                action, hidden, cell, received = agent.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    hidden,
                    cell,
                    received,
                    alive,
                    deterministic=True,
                )
                actions = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = environment.step(actions)
            episode_return += float(reward)
            alive = torch.as_tensor(
                info.get("alive_mask", np.ones(n_agents)), dtype=torch.float32, device=device,
            ).unsqueeze(0)
            if terminated or truncated:
                break
        episode_returns.append(episode_return)
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {
        "returns": episode_returns,
        "successes": successes,
        "mean_distances": distances,
        "communication_rate": float(n_agents > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train faithful recurrent CommNet.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="commnet.pt")
    args = parser.parse_args()
    train(
        env=args.env,
        episodes=args.episodes,
        n_agents=args.n_agents,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
