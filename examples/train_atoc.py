"""Train paper-faithful ATOC on bounded continuous cooperative navigation.

The library learner owns group scheduling, attention supervision, replay, and DDPG.
This script owns environment interaction, the paper's 30-episode warmup, evaluation,
logging, and checkpointing.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl import ATOCConfig, ATOCGroupScheduler, ATOCLearner
from modmarl.algorithms.atoc import PAPER_SOURCE


def train(
    *,
    env: str = "signed_noisy_navigation",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 3_000,
    seed: int = 7,
    config: ATOCConfig | None = None,
    updates_per_episode: int | None = None,
    evaluation_episodes: int = 64,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    """Run ATOC with paper defaults unless an explicit bounded config is supplied."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = _make_environment(env, n_agents, horizon, seed)
    n_agents = environment.n_agents
    architecture = config or ATOCConfig()
    updates_per_episode = environment.horizon if updates_per_episode is None else updates_per_episode
    if updates_per_episode < 0:
        raise ValueError("updates_per_episode must be nonnegative")
    learner = ATOCLearner(
        n_agents,
        environment.obs_dim,
        environment.num_actions,
        architecture,
        action_low=getattr(environment, "action_low", 0.0),
        action_high=getattr(environment, "action_high", 1.0),
        seed=seed,
    ).to(dev)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        learner,
        env,
        n_agents,
        horizon,
        evaluation_seed,
        evaluation_episodes,
        dev,
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
    update_count = 0
    attention_losses: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        scheduler = ATOCGroupScheduler(
            n_agents,
            architecture.communication_period,
            architecture.max_collaborators,
        )
        learner.reset_noise()
        episode_obs, episode_actions, episode_groups = [], [], []
        episode_return = 0.0
        for _ in range(environment.horizon):
            relative, eligible = _communication_candidates(environment, obs)
            output = learner.act(
                torch.as_tensor(obs, dtype=torch.float32, device=dev),
                scheduler,
                np.linalg.norm(relative, axis=-1),
                eligible,
            )
            actions = output.actions.cpu().numpy()
            next_obs, reward, terminated, truncated, info = environment.step(actions)
            rewards = _agent_rewards(reward, info, n_agents)
            learner.store_transition(
                obs,
                actions,
                rewards,
                next_obs,
                np.full(n_agents, terminated, dtype=np.float32),
                output.groups.cpu().numpy(),
            )
            episode_obs.append(torch.as_tensor(obs, dtype=torch.float32, device=dev))
            episode_actions.append(output.actions.detach())
            episode_groups.append(output.groups.detach())
            obs = next_obs
            episode_return += float(np.mean(rewards))

            if terminated or truncated:
                break
        if episode >= architecture.warmup_episodes and episode_obs:
            episode_updates = 0
            for _ in range(updates_per_episode):
                if learner.update() is not None:
                    update_count += 1
                    episode_updates += 1
            if episode_updates:
                attention_loss = learner.update_attention_episode(
                    torch.stack(episode_obs),
                    torch.stack(episode_actions),
                    torch.stack(episode_groups),
                )
                if attention_loss is not None:
                    attention_losses.append(attention_loss)
        returns.append(episode_return)
        if episode % 100 == 0:
            print(f"episode {episode:5d}  return {episode_return:8.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "learner": learner.state_dict(),
                "actor_optimizer": learner.actor_optimizer.state_dict(),
                "critic_optimizer": learner.critic_optimizer.state_dict(),
                "attention_optimizer": learner.attention_optimizer.state_dict(),
                "config": architecture,
            },
            checkpoint,
        )
    final_evaluation = _evaluate(
        learner,
        env,
        n_agents,
        horizon,
        evaluation_seed,
        evaluation_episodes,
        dev,
    )
    environment.close()
    return {
        "algorithm": "atoc",
        "env": env,
        "episodes": episodes,
        "n_agents": n_agents,
        "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "updates": update_count,
        "attention_updates": len(attention_losses),
        "checkpoint": checkpoint,
        "source_revision": PAPER_SOURCE,
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
            "actor_hidden_dims": list(architecture.actor_hidden_dims),
            "critic_hidden_dims": list(architecture.critic_hidden_dims),
            "attention_hidden_dim": architecture.attention_hidden_dim,
            "channel_hidden_dim": architecture.channel_hidden_dim,
            "communication_period": architecture.communication_period,
            "max_collaborators": architecture.max_collaborators,
            "actor_learning_rate": architecture.actor_learning_rate,
            "critic_learning_rate": architecture.critic_learning_rate,
            "attention_learning_rate": architecture.attention_learning_rate,
            "gamma": architecture.gamma,
            "tau": architecture.tau,
            "batch_size": architecture.batch_size,
            "replay_capacity": architecture.replay_capacity,
            "warmup_episodes": architecture.warmup_episodes,
            "ou_theta": architecture.ou_theta,
            "ou_sigma": architecture.ou_sigma,
            "updates_per_episode": updates_per_episode,
        },
    }


def _communication_candidates(environment, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate_fn = getattr(environment, "communication_candidates", None)
    if callable(candidate_fn):
        return candidate_fn()
    positions = obs[:, 2:4]
    relative = positions[:, None] - positions[None]
    distances = np.linalg.norm(relative, axis=-1)
    eligible = ~np.eye(environment.n_agents, dtype=bool)
    if environment.n_agents > 4:
        nearest = np.argsort(distances, axis=1)[:, 1:4]
        eligible.fill(False)
        eligible[np.arange(environment.n_agents)[:, None], nearest] = True
    return relative.astype(np.float32), eligible


def _make_environment(env: str, n_agents: int, horizon: int, seed: int):
    return make_env(env, n_agents, horizon, seed)


def _agent_rewards(reward, info: dict, n_agents: int) -> np.ndarray:
    rewards = np.asarray(info.get("agent_rewards", reward), dtype=np.float32)
    if rewards.ndim == 0:
        rewards = np.full(n_agents, float(rewards), dtype=np.float32)
    if rewards.shape != (n_agents,):
        raise ValueError("agent rewards must have shape (n_agents,)")
    return rewards


@torch.no_grad()
def _evaluate(*args, **kwargs):
    """Evaluate without advancing the training NumPy or Torch random streams."""
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        return _evaluate_impl(*args, **kwargs)
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _evaluate_impl(
    learner,
    env_name,
    n_agents,
    horizon,
    seed,
    episodes,
    device,
    *,
    random_policy=False,
):
    environment = _make_environment(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes, distances, links = [], [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        scheduler = ATOCGroupScheduler(
            environment.n_agents,
            learner.config.communication_period,
            learner.config.max_collaborators,
        )
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                actions = generator.uniform(
                    getattr(environment, "action_low", 0.0),
                    getattr(environment, "action_high", 1.0),
                    size=(environment.n_agents, environment.num_actions),
                )
            else:
                relative, eligible = _communication_candidates(environment, obs)
                output = learner.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device),
                    scheduler,
                    np.linalg.norm(relative, axis=-1),
                    eligible,
                    deterministic=True,
                    explore=False,
                )
                actions = output.actions.cpu().numpy()
                groups = output.groups.cpu().numpy()
                links.extend(groups[~np.eye(environment.n_agents, dtype=bool)].tolist())
            obs, reward, terminated, truncated, info = environment.step(actions)
            episode_return += float(np.mean(_agent_rewards(reward, info, environment.n_agents)))
            if terminated or truncated:
                break
        returns.append(episode_return)
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    environment.close()
    return {
        "returns": returns,
        "successes": successes,
        "mean_distances": distances,
        "communication_rate": float(np.mean(links)) if links else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ATOC on cooperative navigation.")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="atoc.pt")
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        n_agents=args.n_agents,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
