"""Train IPPO on a cooperative MPE task and save a checkpoint.

Run:
    python examples/train_ippo.py --episodes 300

The algorithm file owns IPPO's actor, local critic, GAE rollout, clipped losses,
optimiser, and update.  This example only interacts with the environment, supplies
correct termination bootstraps, evaluates held-out seeds, and saves complete state.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.ippo import IPPOAgent, IPPORollout


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    hidden_dims: tuple[int, int] = (256, 128),
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    critic_coef: float = 1.0,
    entropy_coef: float = 0.005,
    ppo_epochs: int = 4,
    rollout_episodes: int = 8,
    minibatch_size: int = 1024,
    evaluation_episodes: int = 32,
    evaluation_seed: int | None = None,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    n = environment.n_agents
    agent = IPPOAgent(
        obs_dim=environment.obs_dim,
        action_dim=environment.num_actions,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        clip_epsilon=clip_eps,
        critic_coef=critic_coef,
        entropy_coef=entropy_coef,
    ).to(dev)
    evaluation_seed = seed + 100_000 if evaluation_seed is None else evaluation_seed
    initial_evaluation = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    random_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev, random_policy=True,
    )

    returns: list[float] = []
    episode_count = 0
    iteration = 0
    while episode_count < episodes:
        rollout = IPPORollout(gamma, gae_lambda)
        episodes_this_iter = min(rollout_episodes, episodes - episode_count)
        for _ in range(episodes_this_iter):
            obs, _ = environment.reset(seed=seed + episode_count)
            episode_return = 0.0
            terminated = truncated = False
            next_obs = obs
            for _ in range(environment.horizon):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)     # (n, obs_dim)
                available_actions = torch.ones(
                    n, environment.num_actions, dtype=torch.bool, device=dev,
                )
                with torch.no_grad():
                    action, log_prob, value = agent.act(obs_t, available_actions)
                next_obs, reward, terminated, truncated, _ = environment.step(action.cpu().numpy())
                rollout.add(obs_t, available_actions, action, log_prob, value, reward)
                obs = next_obs
                episode_return += reward
                if terminated or truncated:
                    break

            # Bootstrap the truncated tail with each agent's V(o_T); termination bootstraps 0.
            if terminated:
                bootstrap = torch.zeros(n, device=dev)
            else:
                with torch.no_grad():
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
                    bootstrap = agent.critic(next_obs_t.unsqueeze(0)).squeeze(0)
            rollout.finish_episode(bootstrap)
            returns.append(episode_return)
            episode_count += 1

        agent.update(rollout, epochs=ppo_epochs, minibatch_size=minibatch_size)

        if iteration % 5 == 0:
            recent = float(np.mean(returns[-episodes_this_iter:]))
            print(f"iter {iteration:4d}  return {recent:7.2f}", flush=True)
        iteration += 1

    if checkpoint is not None:
        torch.save({
            "model": agent.state_dict(), "optimizer": agent.optimizer.state_dict(),
            "metadata": {
                "schema_version": 1, "algorithm": "ippo", "env": env,
                "n_agents": n, "horizon": environment.horizon,
                "constructor": {"obs_dim": environment.obs_dim, "action_dim": environment.num_actions,
                                "hidden_dims": hidden_dims, "learning_rate": learning_rate,
                                "clip_epsilon": clip_eps, "critic_coef": critic_coef,
                                "entropy_coef": entropy_coef},
            },
        }, checkpoint)

    final_evaluation = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    return {
        "algorithm": "ippo",
        "env": env,
        "episodes": episode_count,
        "n_agents": n,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "validation_criterion": {
            "return_margin_over_random": 2.0,
            "maximum_mean_distance": 0.85,
            "scope": "every confirmation seed",
        },
    }


def _evaluate(agent, env_name, n_agents, horizon, seed, episodes, device, *, random_policy=False):
    """Deterministic held-out evaluation plus a matched random-policy reference."""
    evaluation_env = make_env(env_name, n_agents, horizon, seed)
    returns, successes, distances = [], [], []
    random_generator = np.random.default_rng(seed)
    for episode in range(episodes):
        obs, _ = evaluation_env.reset(seed=seed + episode)
        episode_return, info = 0.0, {}
        for _ in range(evaluation_env.horizon):
            if random_policy:
                action = random_generator.integers(evaluation_env.num_actions, size=evaluation_env.n_agents)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = agent.act(obs_tensor, deterministic=True)[0].cpu().numpy()
            obs, reward, terminated, truncated, info = evaluation_env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {"returns": returns, "successes": successes, "mean_distances": distances}

def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPPO on a cooperative MPE task.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="ippo.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
