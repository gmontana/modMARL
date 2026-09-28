"""Train the released MAGIC policy with its original on-policy objective.

Run:
    python examples/train_magic.py --episodes 300

Complete episodes are accumulated to the release's 500-transition single-worker
batch. Each update recomputes the recurrent policy with the collected Gumbel
perturbations and applies one RMSProp step; there are no PPO epochs or replay.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.magic import MAGICAgent, MAGICConfig


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    architecture: MAGICConfig | None = None,
    batch_steps: int = 500,
    evaluation_episodes: int = 32,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    n_agents = environment.n_agents
    agent = MAGICAgent(
        environment.obs_dim,
        environment.num_actions,
        architecture,
    ).to(dev)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        agent,
        env,
        n_agents,
        horizon,
        evaluation_seed,
        evaluation_episodes,
        dev,
        random_policy=True,
    )

    returns: list[float] = []
    episode_count = 0
    while episode_count < episodes:
        episodes_batch = []
        transition_count = 0
        while transition_count < batch_steps and episode_count < episodes:
            episode, episode_return = _collect_episode(
                agent, environment, n_agents, seed + episode_count, dev,
            )
            episodes_batch.append(episode)
            transition_count += episode["obs"].shape[0]
            returns.append(episode_return)
            episode_count += 1
        tensors = _pad_episodes(
            episodes_batch, n_agents, environment.obs_dim, dev,
        )
        agent.update(**tensors)
        if episode_count % 20 < len(episodes_batch):
            print(f"episodes {episode_count:4d}  return {returns[-1]:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {"agent": agent.state_dict(), "optimizer": agent.optimizer.state_dict()}, checkpoint,
        )
    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    cfg = agent.config
    return {
        "algorithm": "magic",
        "env": env,
        "episodes": episode_count,
        "n_agents": n_agents,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "checkpoint": checkpoint,
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
            "architecture": "released_predator_prey_medium",
            "hidden_dim": cfg.hidden_dim,
            "gat_hidden_dim": cfg.gat_hidden_dim,
            "gat_heads": cfg.gat_heads,
            "directed": cfg.directed,
            "use_gat_encoder": cfg.use_gat_encoder,
            "learn_second_graph": cfg.learn_second_graph,
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "mean_ratio": cfg.mean_ratio,
            "value_coefficient": cfg.value_coefficient,
            "entropy_coefficient": cfg.entropy_coefficient,
            "detach_gap": cfg.detach_gap,
            "batch_steps": batch_steps,
            "optimizer": "RMSprop",
            "rmsprop_alpha": 0.97,
            "rmsprop_epsilon": 1e-6,
            "reward_contract": "per_agent",
        },
    }


@torch.no_grad()
def _collect_episode(agent, environment, n_agents, episode_seed, device):
    obs, _ = environment.reset(seed=episode_seed)
    hidden, cell = agent.init_state(1, n_agents, device)
    alive = np.ones(n_agents, dtype=np.float32)
    steps = {
        key: []
        for key in ("obs", "actions", "rewards", "alive", "continuation", "noise")
    }
    episode_return = 0.0
    for _ in range(environment.horizon):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        alive_tensor = torch.as_tensor(alive, device=device).unsqueeze(0)
        action, _, _, hidden, cell, _, noise = agent.act(
            obs_tensor.unsqueeze(0), hidden, cell, alive=alive_tensor,
        )
        next_obs, reward, terminated, truncated, info = environment.step(
            action.squeeze(0).cpu().numpy(),
        )
        rewards = _agent_rewards(reward, info, n_agents)
        next_alive = _next_alive(info, n_agents)
        continuation = next_alive.copy()
        if terminated or truncated:
            continuation.fill(0.0)
        steps["obs"].append(obs_tensor)
        steps["actions"].append(action.squeeze(0))
        steps["rewards"].append(torch.as_tensor(rewards, device=device))
        steps["alive"].append(alive_tensor.squeeze(0))
        steps["continuation"].append(torch.as_tensor(continuation, device=device))
        steps["noise"].append(noise.squeeze(1))
        obs = next_obs
        alive = next_alive
        episode_return += float(np.mean(rewards))
        if terminated or truncated:
            break
    return {key: torch.stack(values) for key, values in steps.items()}, episode_return


def _agent_rewards(reward, info, n_agents) -> np.ndarray:
    """Preserve released per-agent rewards; expand only a shared scalar."""
    if "agent_rewards" in info:
        rewards = np.asarray(info["agent_rewards"], dtype=np.float32)
    else:
        rewards = np.asarray(reward, dtype=np.float32)
        if rewards.ndim == 0:
            rewards = np.full(n_agents, float(rewards), dtype=np.float32)
    if rewards.shape != (n_agents,):
        raise ValueError(f"agent rewards must have shape {(n_agents,)}, got {rewards.shape}")
    return rewards


def _next_alive(info, n_agents) -> np.ndarray:
    if "alive_mask" not in info:
        return np.ones(n_agents, dtype=np.float32)
    alive = np.asarray(info["alive_mask"], dtype=np.float32)
    if alive.shape != (n_agents,):
        raise ValueError(f"alive_mask must have shape {(n_agents,)}, got {alive.shape}")
    return alive


def _pad_episodes(episodes, n_agents, obs_dim, device):
    batch_size = len(episodes)
    horizon = max(episode["obs"].shape[0] for episode in episodes)
    obs = torch.zeros(batch_size, horizon, n_agents, obs_dim, device=device)
    actions = torch.zeros(batch_size, horizon, n_agents, dtype=torch.long, device=device)
    rewards = torch.zeros(batch_size, horizon, n_agents, device=device)
    mask = torch.zeros_like(rewards)
    alive = torch.zeros_like(rewards)
    continuation = torch.zeros_like(rewards)
    noise = torch.zeros(batch_size, horizon, 2, n_agents, n_agents, 2, device=device)
    for index, episode in enumerate(episodes):
        length = episode["obs"].shape[0]
        obs[index, :length] = episode["obs"]
        actions[index, :length] = episode["actions"]
        rewards[index, :length] = episode["rewards"]
        mask[index, :length] = 1.0
        alive[index, :length] = episode["alive"]
        continuation[index, :length] = episode["continuation"]
        noise[index, :length] = episode["noise"]
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "mask": mask,
        "alive": alive,
        "continuation": continuation,
        "noise": noise,
    }


@torch.no_grad()
def _evaluate(
    agent,
    env_name,
    n_agents,
    horizon,
    seed,
    episodes,
    device,
    *,
    random_policy: bool = False,
):
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes, distances, edges = [], [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        hidden, cell = agent.init_state(1, n_agents, device)
        alive = np.ones(n_agents, dtype=np.float32)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                action = generator.integers(environment.num_actions, size=n_agents)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                alive_tensor = torch.as_tensor(alive, device=device).unsqueeze(0)
                replay_noise = torch.zeros(
                    2, 1, n_agents, n_agents, 2, device=device,
                )
                logits, _, hidden, cell, adjacency, _ = agent(
                    obs_tensor,
                    hidden,
                    cell,
                    alive=alive_tensor,
                    noise=replay_noise,
                )
                action = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                edges.extend(adjacency.cpu().reshape(-1).tolist())
            obs, reward, terminated, truncated, info = environment.step(action)
            rewards = _agent_rewards(reward, info, n_agents)
            episode_return += float(np.mean(rewards))
            alive = _next_alive(info, n_agents)
            if terminated or truncated:
                break
        returns.append(episode_return)
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {
        "returns": returns,
        "successes": successes,
        "mean_distances": distances,
        "communication_rate": float(np.mean(edges)) if edges else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train faithful MAGIC.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="magic.pt")
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
