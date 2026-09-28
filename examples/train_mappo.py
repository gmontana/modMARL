"""Train the self-contained recurrent MAPPO implementation on an MPE-style task.

The algorithm module owns the networks, recurrent rollout/chunk representation,
GAE, normalization, objectives, optimizers, and update.  This runner only adapts
the environment's observations to the paper's MPE centralized critic input,
collects episodes, performs held-out evaluation, and saves complete state.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.mappo import MAPPOAgent, MAPPORollout


def _centralized_states(obs: np.ndarray) -> np.ndarray:
    """Paper MPE CL state: concatenate all observations and repeat per agent."""
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.repeat(flat[None], obs.shape[0], axis=0)


def train(
    *, env: str = "navigation", n_agents: int = 3, horizon: int = 25,
    episodes: int = 300, seed: int = 7, hidden_dim: int = 64,
    actor_lr: float = 7e-4, critic_lr: float = 7e-4,
    gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
    entropy_coef: float = 0.01, ppo_epochs: int = 10,
    rollout_episodes: int = 128, num_minibatches: int = 1,
    chunk_length: int = 10, evaluation_episodes: int = 32,
    device: str = "cpu", checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    learner = MAPPOAgent(
        environment.n_agents, environment.obs_dim, environment.num_actions,
        state_dim=environment.n_agents * environment.obs_dim, hidden_dim=hidden_dim,
        actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_epsilon=clip_eps, entropy_coef=entropy_coef, chunk_length=chunk_length,
    ).to(dev)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    random_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev, random_policy=True,
    )

    returns: list[float] = []
    episode_count = 0
    while episode_count < episodes:
        rollout = MAPPORollout(learner)
        count = min(rollout_episodes, episodes - episode_count)
        for _ in range(count):
            obs, _ = environment.reset(seed=seed + episode_count)
            actor_hidden, critic_hidden = learner.initial_state(dev)
            mask = torch.ones(environment.n_agents, device=dev)
            episode_return = 0.0
            terminated = False
            for _ in range(environment.horizon):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                states = torch.as_tensor(_centralized_states(obs), device=dev)
                previous_actor, previous_critic = actor_hidden, critic_hidden
                actions, log_probs, values, actor_hidden, critic_hidden = learner.act(
                    obs_tensor, states, actor_hidden, critic_hidden, mask,
                )
                next_obs, reward, terminated, truncated, _ = environment.step(actions.cpu().numpy())
                rollout.add(
                    obs=obs_tensor, states=states, actions=actions, log_probs=log_probs,
                    values=values, actor_hidden=previous_actor, critic_hidden=previous_critic,
                    masks=mask, team_reward=reward,
                )
                obs = next_obs
                episode_return += reward
                if terminated or truncated:
                    break
            final_mask = torch.zeros(environment.n_agents, device=dev) if terminated else mask
            if terminated:
                bootstrap = torch.zeros(environment.n_agents, device=dev)
            else:
                with torch.no_grad():
                    states = torch.as_tensor(_centralized_states(obs), device=dev)
                    bootstrap, _ = learner.critic.step(states, critic_hidden, mask)
            rollout.finish_episode(bootstrap, final_mask)
            returns.append(float(episode_return))
            episode_count += 1
        learner.update(rollout.batch(), epochs=ppo_epochs, num_minibatches=num_minibatches)

    if checkpoint is not None:
        torch.save(
            {"model": learner.state_dict(), "actor_optimizer": learner.actor_optimizer.state_dict(),
             "critic_optimizer": learner.critic_optimizer.state_dict(), "episodes": episode_count},
            checkpoint,
        )
    final_evaluation = _evaluate(learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    return {
        "algorithm": "mappo", "env": env, "episodes": episode_count,
        "n_agents": environment.n_agents, "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0, "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation, "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "validation_criterion": {
            "return_margin_over_random": 5.0,
            "maximum_mean_distance": 0.60,
            "scope": "every confirmation seed",
        },
    }


def _evaluate(learner, env_name, n_agents, horizon, seed, episodes, device, *, random_policy=False):
    evaluation_env = make_env(env_name, n_agents, horizon, seed)
    returns, successes, distances = [], [], []
    random_generator = np.random.default_rng(seed)
    for episode in range(episodes):
        obs, _ = evaluation_env.reset(seed=seed + episode)
        actor_hidden, critic_hidden = learner.initial_state(device)
        mask = torch.ones(evaluation_env.n_agents, device=device)
        episode_return, info = 0.0, {}
        for _ in range(evaluation_env.horizon):
            if random_policy:
                actions = random_generator.integers(evaluation_env.num_actions, size=evaluation_env.n_agents)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                states = torch.as_tensor(_centralized_states(obs), device=device)
                actions, _, _, actor_hidden, critic_hidden = learner.act(
                    obs_tensor, states, actor_hidden, critic_hidden, mask, deterministic=True,
                )
                actions = actions.cpu().numpy()
            obs, reward, terminated, truncated, info = evaluation_env.step(actions)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {"returns": returns, "successes": successes, "mean_distances": distances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recurrent MAPPO.")
    parser.add_argument("--env", default="navigation")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="mappo.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents,
          seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
