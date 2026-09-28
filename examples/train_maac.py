"""Train the complete MAAC learner on a cooperative discrete-action task.

The example owns environment interaction, deterministic evaluation, and checkpoint
serialization. Model, replay, optimizer, entropy target, and update cadence remain in
``MAACLearner`` so library users do not need to reconstruct the algorithm.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl import MAACConfig, MAACLearner


def train(
    *,
    env: str = "paper_navigation",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    pi_lr: float = 1e-3,
    q_lr: float = 1e-3,
    hidden_dim: int = 128,
    attend_heads: int = 4,
    critic_weight_decay: float = 1e-3,
    buffer_size: int = 1_000_000,
    batch_size: int = 1024,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    config = MAACConfig(
        policy_learning_rate=pi_lr,
        critic_learning_rate=q_lr,
        critic_weight_decay=critic_weight_decay,
        replay_capacity=buffer_size,
        batch_size=batch_size,
    )
    learner = MAACLearner(
        environment.n_agents,
        environment.obs_dim,
        environment.num_actions,
        hidden_dim=hidden_dim,
        attend_heads=attend_heads,
        config=config,
    ).to(dev)

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
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        for _ in range(environment.horizon):
            action = learner.act(
                torch.as_tensor(obs, dtype=torch.float32, device=dev),
            ).cpu().numpy()
            next_obs, reward, terminated, truncated, info = environment.step(action)
            learner.store_transition(
                obs, action, info.get("agent_rewards", reward), next_obs, terminated,
            )
            learner.update()
            obs = next_obs
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "learner": learner.state_dict(),
                "actor_optimizers": [
                    optimizer.state_dict() for optimizer in learner.actor_optimizers
                ],
                "critic_optimizer": learner.critic_optimizer.state_dict(),
                "total_steps": learner.total_steps,
            },
            checkpoint,
        )

    final_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    return {
        "algorithm": "maac",
        "env": env,
        "episodes": episodes,
        "n_agents": environment.n_agents,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "validation_criterion": {
            "return_improvement_fraction_of_abs_random": 0.2,
            "scope": "every confirmation seed",
        },
        "config": {
            "env": env,
            "episodes": episodes,
            "seed": seed,
            "n_agents": environment.n_agents,
            "horizon": environment.horizon,
            "gamma": config.gamma,
            "tau": config.tau,
            "entropy_temperature": config.entropy_temperature,
            "pi_lr": config.policy_learning_rate,
            "q_lr": config.critic_learning_rate,
            "critic_weight_decay": config.critic_weight_decay,
            "batch_size": config.batch_size,
            "buffer_size": config.replay_capacity,
            "update_interval": config.update_interval,
            "updates_per_interval": config.updates_per_interval,
            "hidden_dim": hidden_dim,
            "attend_heads": attend_heads,
        },
    }


@torch.no_grad()
def _evaluate(
    learner: MAACLearner,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    random_policy: bool = False,
) -> dict:
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes = [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            action = (
                generator.integers(
                    0, environment.num_actions, size=environment.n_agents,
                )
                if random_policy
                else learner.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device),
                    deterministic=True,
                ).cpu().numpy()
            )
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
    return {"returns": returns, "successes": successes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAAC on a cooperative task.")
    parser.add_argument("--env", default="paper_navigation")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="maac.pt")
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
