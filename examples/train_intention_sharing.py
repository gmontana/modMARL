"""Train the complete Intention Sharing learner on Cooperative Navigation.

The example owns environment interaction, deterministic evaluation, and checkpoint
serialization. Independent actors and critics, recurrent message replay, predictor
losses, and target updates remain inside ``IntentionSharingLearner``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from marl_envs import make_env
from modmarl import IntentionSharingConfig, IntentionSharingLearner


def train(
    *,
    env: str = "paper_navigation",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    learning_rate: float = 5e-4,
    hidden_dim: int = 128,
    message_dim: int = 3,
    imagination_horizon: int = 5,
    buffer_size: int = 200_000,
    batch_size: int = 128,
    warmup_steps: int = 1_000,
    update_interval: int = 100,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    config = IntentionSharingConfig(
        learning_rate=learning_rate,
        replay_capacity=buffer_size,
        batch_size=batch_size,
        minimum_replay_size=warmup_steps,
    )
    learner = IntentionSharingLearner(
        environment.n_agents,
        environment.obs_dim,
        environment.num_actions,
        message_dim=message_dim,
        hidden_dim=hidden_dim,
        imagination_horizon=imagination_horizon,
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

    total_steps = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        messages = torch.zeros(1, environment.n_agents, message_dim, device=dev)
        episode_return = 0.0
        for _ in range(environment.horizon):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                action_one_hot, action_index, _, written = learner.sample(obs_tensor, messages)
                if total_steps < warmup_steps:
                    action_index = torch.randint(
                        environment.num_actions,
                        (1, environment.n_agents),
                        device=dev,
                    )
                    action_one_hot = F.one_hot(
                        action_index, environment.num_actions,
                    ).to(dtype=obs_tensor.dtype)
                    written = learner.messages_for_actions(
                        obs_tensor, messages, action_one_hot,
                    )
            action = action_index.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = environment.step(action)
            learner.store(
                obs=obs,
                actions=action,
                rewards=info.get("agent_rewards", reward),
                next_obs=next_obs,
                dones=np.full(environment.n_agents, terminated),
                messages_seen=messages.squeeze(0).cpu().numpy(),
                messages_written=written.squeeze(0).cpu().numpy(),
            )
            obs = next_obs
            messages = written
            episode_return += reward
            total_steps += 1
            if learner.ready() and total_steps % update_interval == 0:
                learner.update()
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "learner": learner.state_dict(),
                "policy_optimizers": [
                    optimizer.state_dict() for optimizer in learner.policy_optimizers
                ],
                "critic_optimizers": [
                    optimizer.state_dict() for optimizer in learner.critic_optimizers
                ],
                "total_steps": total_steps,
            },
            checkpoint,
        )

    final_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    return {
        "algorithm": "intention_sharing",
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
            "learning_rate": config.learning_rate,
            "model_loss_weight": config.model_loss_weight,
            "policy_regularization": config.policy_regularization,
            "batch_size": config.batch_size,
            "buffer_size": config.replay_capacity,
            "warmup_steps": config.minimum_replay_size,
            "update_interval": update_interval,
            "message_dim": message_dim,
            "hidden_dim": hidden_dim,
            "imagination_horizon": imagination_horizon,
            "parameter_sharing": False,
        },
    }


@torch.no_grad()
def _evaluate(
    learner: IntentionSharingLearner,
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
        messages = torch.zeros(1, environment.n_agents, learner.message_dim, device=device)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                action = generator.integers(
                    0, environment.num_actions, size=environment.n_agents,
                )
            else:
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=device,
                ).unsqueeze(0)
                _, action_index, _, messages = learner.sample(
                    obs_tensor, messages, deterministic=True,
                )
                action = action_index.squeeze(0).cpu().numpy()
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
    parser = argparse.ArgumentParser(description="Train Intention Sharing.")
    parser.add_argument("--env", default="paper_navigation")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="intention_sharing.pt")
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
