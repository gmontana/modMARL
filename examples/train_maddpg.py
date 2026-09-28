"""Train paper-faithful discrete MADDPG on an MPE task.

The algorithm, optimizers, and complete Eq. 5--6 update live in
``modmarl.algorithms.maddpg``.  This file owns only environment interaction,
the released update schedule, logging, and checkpointing.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl import MADDPGConfig, MADDPGLearner


def train(
    *,
    env: str = "paper_maddpg_navigation",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 3000,
    seed: int = 7,
    learning_rate: float = 1e-2,
    hidden_dim: int = 64,
    buffer_size: int = 1_000_000,
    batch_size: int = 1024,
    warmup_steps: int | None = None,
    update_interval: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
    evaluation_episodes: int = 100,
    evaluation_seed: int = 20_001,
) -> dict:
    """Run MADDPG using the released OpenAI defaults and update cadence."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    n = environment.n_agents
    config = MADDPGConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        replay_capacity=buffer_size,
        max_episode_len=environment.horizon,
        update_interval=update_interval,
        minimum_replay_size=warmup_steps,
    )
    learner = MADDPGLearner(n, environment.obs_dim, environment.num_actions, hidden_dim, config).to(dev)
    initial_evaluation = _evaluate(
        learner, env, n, horizon, evaluation_seed, evaluation_episodes, dev
    )
    random_evaluation = _evaluate(
        learner, env, n, horizon, evaluation_seed, evaluation_episodes, dev, random_actions=True
    )

    total_steps = 0
    returns: list[float] = []
    update_count = 0
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        for _ in range(environment.horizon):
            action_vectors, action_indices = _select_actions(learner, obs, dev)
            env_action = action_vectors if env.startswith("paper_") else action_indices
            next_obs, reward, terminated, truncated, _ = environment.step(env_action)
            learner.store_transition(obs, action_vectors, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            total_steps += 1

            if learner.update():
                update_count += 1
            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 100 == 0:
            print(f"episode {episode:5d}  return {episode_return:8.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "learner": learner.state_dict(),
                "actor_optimizers": [agent.actor_optimizer.state_dict() for agent in learner.agents],
                "critic_optimizers": [agent.critic_optimizer.state_dict() for agent in learner.agents],
                "config": config,
                "steps": total_steps,
            },
            checkpoint,
        )
    final_evaluation = _evaluate(
        learner, env, n, horizon, evaluation_seed, evaluation_episodes, dev
    )
    environment.close()
    return {
        "algorithm": "maddpg",
        "env": env,
        "episodes": episodes,
        "n_agents": n,
        "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "updates": update_count,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "validation_criterion": {
            "return_margin_over_initial": 3.0,
            "scope": "mean deterministic held-out return for every confirmation seed",
        },
        "checkpoint": checkpoint,
    }


@torch.no_grad()
def _select_actions(
    learner: MADDPGLearner, obs: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the reference soft actions and derive indices for the local env API."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions, indices = learner.act(obs_t)
    return actions.cpu().numpy(), indices.cpu().numpy()


def _evaluate(
    learner: MADDPGLearner,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    random_actions: bool = False,
) -> dict[str, list[float]]:
    """Evaluate on held-out resets without perturbing the training RNG streams."""
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    returns, distances, successes = [], [], []
    try:
        environment = make_env(env_name, n_agents, horizon, seed)
        rng = np.random.default_rng(seed)
        for episode in range(episodes):
            obs, _ = environment.reset(seed=seed + episode)
            episode_return = 0.0
            info = {}
            for _ in range(environment.horizon):
                if random_actions:
                    action = rng.integers(environment.num_actions, size=environment.n_agents)
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    vectors, indices = learner.act(obs_t, deterministic=True)
                    action = vectors.cpu().numpy() if env_name.startswith("paper_") else indices.cpu().numpy()
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
            distances.append(float(info.get("mean_distance", float("nan"))))
            successes.append(float(bool(info.get("success", False))))
        environment.close()
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    return {"returns": returns, "mean_distances": distances, "successes": successes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MADDPG on the paper MPE navigation task.")
    parser.add_argument("--env", default="paper_maddpg_navigation")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="maddpg.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
