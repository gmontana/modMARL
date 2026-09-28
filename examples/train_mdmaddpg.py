"""Train the complete MD-MADDPG learner on cooperative navigation.

The example owns environment interaction, deterministic evaluation, and checkpoint
serialization. The algorithm module owns sequential memory, replay, optimizers, and
the archived update schedule.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl import MDMADDPGConfig, MDMADDPGLearner


def train(
    *,
    env: str = "paper_mdmaddpg_navigation",
    n_agents: int = 2,
    horizon: int = 100,
    episodes: int = 300,
    seed: int = 7,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    memory_dim: int = 200,
    buffer_size: int = 1_000_000,
    batch_size: int = 1024,
    update_every: int = 100,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    config = MDMADDPGConfig(
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        memory_dim=memory_dim,
        replay_capacity=buffer_size,
        batch_size=batch_size,
        update_interval=update_every,
    )
    learner = MDMADDPGLearner(
        environment.n_agents,
        environment.obs_dim,
        environment.num_actions,
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
        learner.reset_memory()
        episode_return = 0.0
        for _ in range(environment.horizon):
            action, memory_seen, memory_written = learner.act(
                torch.as_tensor(obs, dtype=torch.float32, device=dev),
            )
            action_array = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = environment.step(action_array)
            learner.store_transition(
                obs,
                action_array,
                info.get("agent_rewards", reward),
                next_obs,
                terminated,
                memory_seen.cpu().numpy(),
                memory_written.cpu().numpy(),
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
                "critic_optimizers": [
                    optimizer.state_dict() for optimizer in learner.critic_optimizers
                ],
                "total_steps": learner.total_steps,
            },
            checkpoint,
        )

    final_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    return {
        "algorithm": "mdmaddpg",
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
        "communication_rate": 1.0,
        "config": {
            "env": env,
            "episodes": episodes,
            "seed": seed,
            "n_agents": environment.n_agents,
            "horizon": environment.horizon,
            "gamma": config.gamma,
            "tau": config.tau,
            "actor_lr": config.actor_learning_rate,
            "critic_lr": config.critic_learning_rate,
            "memory_dim": config.memory_dim,
            "buffer_size": config.replay_capacity,
            "batch_size": config.batch_size,
            "update_interval": config.update_interval,
        },
    }


@torch.no_grad()
def _evaluate(
    learner: MDMADDPGLearner,
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
        learner.reset_memory()
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
                )[0].cpu().numpy()
            )
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
    return {
        "returns": returns,
        "successes": successes,
        "communication_rate": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MD-MADDPG.")
    parser.add_argument("--env", default="paper_mdmaddpg_navigation")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="mdmaddpg.pt")
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
