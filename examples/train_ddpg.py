"""Train independent continuous DDPG learners on cooperative navigation.

Run:
    python examples/train_ddpg.py --episodes 300

Each agent owns the complete paper learner and receives only its local observation.
The trainer owns continuous replay, OU episode resets, evaluation, and checkpoints.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs.noisy_navigation import NoisyNavigationEnv
from modmarl import DDPGAgent, DDPGConfig, DDPGReplayBuffer


def train(
    *,
    env: str = "noisy_navigation",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    hidden_dims: tuple[int, int] = (400, 300),
    buffer_size: int = 1_000_000,
    batch_size: int = 64,
    actor_learning_rate: float = 1e-4,
    critic_learning_rate: float = 1e-3,
    critic_weight_decay: float = 1e-2,
    evaluation_episodes: int = 32,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    if env != "noisy_navigation":
        raise ValueError("faithful DDPG requires a bounded continuous-action environment")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = NoisyNavigationEnv(n_agents=n_agents, horizon=horizon, seed=seed)
    config = DDPGConfig(
        batch_size=batch_size,
        replay_capacity=buffer_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        critic_weight_decay=critic_weight_decay,
    )
    agents = [
        DDPGAgent(
            obs_dim=environment.obs_dim,
            action_dim=environment.num_actions,
            hidden_dims=hidden_dims,
            action_low=0.0,
            action_high=1.0,
            config=config,
            seed=seed + i,
        ).to(dev)
        for i in range(n_agents)
    ]
    replay = DDPGReplayBuffer(
        buffer_size, n_agents, environment.obs_dim, environment.num_actions,
    )
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agents, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        agents, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        random_policy=True,
    )

    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        for agent in agents:
            agent.reset_noise()
        episode_return = 0.0
        for _ in range(environment.horizon):
            actions = _select_actions(agents, obs, dev, explore=True)
            next_obs, reward, terminated, truncated, _ = environment.step(actions)
            replay.add(obs, actions, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward

            if config.update_due(len(replay)):
                _update(agents, replay.sample(batch_size, dev))
            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "agents": [agent.state_dict() for agent in agents],
                "actor_optimizers": [agent.actor_optimizer.state_dict() for agent in agents],
                "critic_optimizers": [agent.critic_optimizer.state_dict() for agent in agents],
            },
            checkpoint,
        )

    final_evaluation = _evaluate(
        agents, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    return {
        "algorithm": "ddpg",
        "env": env,
        "episodes": episodes,
        "n_agents": n_agents,
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
            "hidden_dims": list(hidden_dims),
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "critic_weight_decay": critic_weight_decay,
            "gamma": config.gamma,
            "tau": config.tau,
            "ou_theta": config.ou_theta,
            "ou_sigma": config.ou_sigma,
        },
    }


@torch.no_grad()
def _select_actions(agents, obs, device, *, explore: bool) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return np.stack(
        [
            agent.act(obs_tensor[i].unsqueeze(0), explore=explore).squeeze(0).cpu().numpy()
            for i, agent in enumerate(agents)
        ],
        axis=0,
    )


def _update(agents, batch):
    """Route each local continuous replay slice into its independent learner."""
    return [
        agent.update(
            batch.obs[:, i],
            batch.actions[:, i],
            batch.rewards,
            batch.next_obs[:, i],
            batch.dones,
        )
        for i, agent in enumerate(agents)
    ]


@torch.no_grad()
def _evaluate(
    agents, n_agents, horizon, seed, episodes, device, *, random_policy: bool = False,
):
    environment = NoisyNavigationEnv(n_agents=n_agents, horizon=horizon, seed=seed)
    generator = np.random.default_rng(seed)
    returns, successes, distances = [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            actions = (
                generator.uniform(0.0, 1.0, size=(n_agents, environment.num_actions))
                if random_policy
                else _select_actions(agents, obs, device, explore=False)
            )
            obs, reward, terminated, truncated, info = environment.step(actions)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {"returns": returns, "successes": successes, "mean_distances": distances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train independent continuous DDPG.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="ddpg.pt")
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        n_agents=args.n_agents,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
