"""Collect trajectories for the complete HAPPO learner and evaluate it."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.happo import HAPPOAgent, HAPPOBatch


def train(
    *, env: str = "navigation", n_agents: int = 3, horizon: int = 25,
    episodes: int = 5000, seed: int = 7, hidden_dim: int = 128,
    actor_lr: float = 5e-4, critic_lr: float = 5e-4,
    gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
    entropy_coef: float = 0.01, ppo_epochs: int = 5,
    rollout_episodes: int = 160, evaluation_episodes: int = 64,
    device: str = "cpu", checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    learner = HAPPOAgent(
        environment.n_agents, environment.obs_dim, environment.num_actions, hidden_dim,
        actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_epsilon=clip_eps, entropy_coef=entropy_coef,
    ).to(dev)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    random_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev, random_policy=True,
    )

    returns: list[float] = []
    episode_count = 0
    while episode_count < episodes:
        pieces: list[HAPPOBatch] = []
        for _ in range(min(rollout_episodes, episodes - episode_count)):
            obs, _ = environment.reset(seed=seed + episode_count)
            trajectory: dict[str, list[torch.Tensor]] = {
                key: [] for key in ("obs", "states", "actions", "old_log_probs", "old_values")
            }
            rewards: list[torch.Tensor] = []
            episode_return = 0.0
            terminated = False
            for _ in range(environment.horizon):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                state = obs_tensor.reshape(-1)
                with torch.no_grad():
                    actions, log_probs = learner.act(obs_tensor)
                    value = learner.critic(state)
                next_obs, reward, terminated, truncated, _ = environment.step(actions.cpu().numpy())
                for key, value_tensor in zip(
                    trajectory, (obs_tensor, state, actions, log_probs, value),
                ):
                    trajectory[key].append(value_tensor.detach())
                rewards.append(torch.as_tensor(float(reward), device=dev))
                obs = next_obs
                episode_return += reward
                if terminated or truncated:
                    break
            with torch.no_grad():
                bootstrap = torch.zeros((), device=dev) if terminated else learner.critic(
                    torch.as_tensor(obs, dtype=torch.float32, device=dev).reshape(-1)
                )
            values = torch.stack(trajectory["old_values"])
            reward_tensor = torch.stack(rewards)
            masks = torch.ones_like(reward_tensor)
            if terminated:
                masks[-1] = 0.0
            advantages, value_targets = learner.compute_gae(reward_tensor, values, bootstrap, masks)
            length = reward_tensor.shape[0]
            pieces.append(HAPPOBatch(
                **{key: torch.stack(value) for key, value in trajectory.items()},
                advantages=advantages, returns=value_targets,
                active_masks=torch.ones(length, environment.n_agents, device=dev),
                available_actions=torch.ones(
                    length, environment.n_agents, environment.num_actions, device=dev, dtype=torch.bool,
                ),
            ))
            returns.append(float(episode_return))
            episode_count += 1
        learner.update(_concatenate(pieces), actor_epochs=ppo_epochs, critic_epochs=ppo_epochs)

    if checkpoint is not None:
        torch.save({
            "model": learner.state_dict(),
            "actor_optimizers": [optimizer.state_dict() for optimizer in learner.actor_optimizers],
            "critic_optimizer": learner.critic_optimizer.state_dict(), "episodes": episode_count,
        }, checkpoint)
    final_evaluation = _evaluate(learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    return {
        "algorithm": "happo", "env": env, "episodes": episode_count,
        "n_agents": environment.n_agents, "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0, "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation, "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "validation_criterion": {
            "return_margin_over_random": 5.0,
            "maximum_mean_distance": 0.80,
            "scope": "every confirmation seed",
        },
    }


def _concatenate(pieces: list[HAPPOBatch]) -> HAPPOBatch:
    return HAPPOBatch(**{
        field: torch.cat([getattr(piece, field) for piece in pieces])
        for field in HAPPOBatch.__dataclass_fields__
    })


@torch.no_grad()
def _evaluate(learner, env_name, n_agents, horizon, seed, episodes, device, *, random_policy=False):
    environment = make_env(env_name, n_agents, horizon, seed)
    returns, successes, distances = [], [], []
    generator = np.random.default_rng(seed)
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                actions = generator.integers(environment.num_actions, size=environment.n_agents)
            else:
                actions = learner.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device), deterministic=True,
                )[0].cpu().numpy()
            obs, reward, terminated, truncated, info = environment.step(actions)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {"returns": returns, "successes": successes, "mean_distances": distances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HAPPO.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="happo.pt")
    args = parser.parse_args()
    train(episodes=args.episodes, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
