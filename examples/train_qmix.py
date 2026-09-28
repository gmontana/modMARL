"""Train paper-faithful recurrent QMIX with centralized environment state.

The runner owns environment adaptation and episode collection only. All QMIX
networks, action selection, double-Q loss, optimizer, and target updates live in
``modmarl.algorithms.qmix``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.qmix import QMIXAgent, QMIXReplayBuffer


def _global_state(environment) -> np.ndarray:
    """Return factual simulator state, never a silent concatenation of observations."""
    state_method = getattr(environment, "state", None)
    if callable(state_method):
        return np.asarray(state_method(), dtype=np.float32).reshape(-1)
    fields = [getattr(environment, name, None) for name in ("positions", "velocities", "landmarks")]
    if all(field is not None for field in fields):
        return np.concatenate([np.asarray(field).reshape(-1) for field in fields]).astype(np.float32)
    raise TypeError(f"{type(environment).__name__} must expose centralized state for QMIX")


def train(
    *,
    env: str = "navigation", n_agents: int = 3, horizon: int = 25,
    episodes: int = 300, seed: int = 7, learning_rate: float = 5e-4,
    hidden_dim: int = 64, mixer_hidden_dim: int = 32, buffer_size: int = 500,
    batch_size: int = 32, warmup_episodes: int = 32, updates_per_episode: int = 1,
    target_update_interval: int = 200, epsilon_anneal_steps: int = 20_000,
    gamma: float = 0.99, evaluation_episodes: int = 32,
    device: str = "cpu", checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    obs, _ = environment.reset(seed=seed)
    state_dim = _global_state(environment).size
    learner = QMIXAgent(
        environment.n_agents, environment.obs_dim, environment.num_actions, state_dim,
        hidden_dim, mixer_hidden_dim, learning_rate=learning_rate, gamma=gamma,
        target_update_interval=target_update_interval,
        epsilon_anneal_steps=epsilon_anneal_steps,
    ).to(dev)
    replay = QMIXReplayBuffer(
        buffer_size, horizon, environment.n_agents, environment.obs_dim,
        state_dim, environment.num_actions,
    )
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(learner, env, n_agents, horizon, evaluation_seed,
                                   evaluation_episodes, dev)
    random_evaluation = _evaluate(learner, env, n_agents, horizon, evaluation_seed,
                                  evaluation_episodes, dev, random_policy=True)
    total_steps = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        learner.reset_hidden(dev)
        observations, states = [obs.copy()], [_global_state(environment)]
        actions, rewards, dones = [], [], []
        availability = [np.ones((environment.n_agents, environment.num_actions), np.float32)]
        episode_return = 0.0
        for _ in range(environment.horizon):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            available_tensor = torch.ones(
                environment.n_agents, environment.num_actions, device=dev,
            )
            action = learner.act(obs_tensor, available_tensor, learner.epsilon(total_steps)).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = environment.step(action)
            observations.append(next_obs.copy())
            states.append(_global_state(environment))
            availability.append(np.ones_like(availability[-1]))
            actions.append(action.copy())
            rewards.append(reward)
            dones.append(float(terminated))
            obs = next_obs
            episode_return += reward
            total_steps += 1
            if terminated or truncated:
                break
        replay.add_episode(
            obs=np.asarray(observations), states=np.asarray(states), actions=np.asarray(actions),
            available_actions=np.asarray(availability), rewards=np.asarray(rewards),
            dones=np.asarray(dones),
        )
        if len(replay) >= max(batch_size, warmup_episodes):
            for _ in range(updates_per_episode):
                learner.update(replay.sample(batch_size, dev))
        learner.maybe_update_targets(episode + 1)
        returns.append(episode_return)

    if checkpoint is not None:
        torch.save({"model": learner.state_dict(), "optimizer": learner.optimizer.state_dict(),
                    "episodes": episodes, "total_steps": total_steps}, checkpoint)
    final_evaluation = _evaluate(learner, env, n_agents, horizon, evaluation_seed,
                                 evaluation_episodes, dev)
    return {
        "algorithm": "qmix", "env": env, "episodes": episodes,
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
    cuda_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        random_generator = np.random.default_rng(seed)
        for episode in range(episodes):
            obs, _ = evaluation_env.reset(seed=seed + episode)
            learner.reset_hidden(device)
            episode_return, info = 0.0, {}
            for _ in range(evaluation_env.horizon):
                if random_policy:
                    action = random_generator.integers(
                        evaluation_env.num_actions, size=evaluation_env.n_agents,
                    )
                else:
                    action = learner.act(
                        torch.as_tensor(obs, dtype=torch.float32, device=device),
                        torch.ones(evaluation_env.n_agents, evaluation_env.num_actions, device=device),
                        0.0,
                    ).cpu().numpy()
                obs, reward, terminated, truncated, info = evaluation_env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            successes.append(float(bool(info.get("success", False))))
            distances.append(float(info.get("mean_distance", float("nan"))))
    return {"returns": returns, "successes": successes, "mean_distances": distances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recurrent QMIX.")
    parser.add_argument("--env", default="navigation")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="qmix.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents,
          seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
