"""Train IC3Net with individualized REINFORCE and learned communication gates.

Run:
    python examples/train_ic3net.py --episodes 300

Complete episodes are accumulated to the released 500-transition update batch. The
environment may expose ``info['agent_rewards']``; otherwise a cooperative scalar is
expanded without changing the learner's per-agent reward contract.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.ic3net import IC3NetAgent


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    hidden_dim: int = 128,
    learning_rate: float = 1e-3,
    gamma: float = 1.0,
    value_coefficient: float = 0.01,
    entropy_coefficient: float = 0.0,
    detach_gap: int = 10,
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
    agent = IC3NetAgent(
        environment.obs_dim,
        environment.num_actions,
        hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
        detach_gap=detach_gap,
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
        batch = []
        transition_count = 0
        while transition_count < batch_steps and episode_count < episodes:
            episode, episode_return = _collect_episode(
                agent, environment, n_agents, seed + episode_count, dev,
            )
            batch.append(episode)
            transition_count += episode["obs"].shape[0]
            returns.append(episode_return)
            episode_count += 1
        tensors = _pad_episodes(batch, n_agents, environment.obs_dim, dev)
        agent.update(**tensors)
        if episode_count % 20 < len(batch):
            print(f"episodes {episode_count:4d}  return {returns[-1]:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {"agent": agent.state_dict(), "optimizer": agent.optimizer.state_dict()}, checkpoint,
        )
    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    return {
        "algorithm": "ic3net",
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
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "value_coefficient": value_coefficient,
            "entropy_coefficient": entropy_coefficient,
            "detach_gap": detach_gap,
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
    hidden, cell, previous_gate = agent.init_state(1, n_agents, device)
    steps = {
        key: []
        for key in ("obs", "actions", "gates", "rewards", "alive", "continuation")
    }
    episode_return = 0.0
    alive = np.ones(n_agents, dtype=np.float32)
    for _ in range(environment.horizon):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action, _, gate, _, _, hidden, cell = agent.act(
            obs_tensor.unsqueeze(0), hidden, cell, previous_gate,
        )
        next_obs, reward, terminated, truncated, info = environment.step(
            action.squeeze(0).cpu().numpy(),
        )
        steps["obs"].append(obs_tensor)
        steps["actions"].append(action.squeeze(0))
        steps["gates"].append(gate.squeeze(0))
        steps["rewards"].append(
            torch.as_tensor(_agent_rewards(reward, info, n_agents), device=device),
        )
        next_alive = _agent_alive(info, n_agents)
        continuation = next_alive.copy()
        if terminated or truncated:
            continuation.fill(0.0)
        steps["alive"].append(torch.as_tensor(alive, device=device))
        steps["continuation"].append(torch.as_tensor(continuation, device=device))
        previous_gate = gate
        obs = next_obs
        alive = next_alive
        episode_return += float(np.mean(_agent_rewards(reward, info, n_agents)))
        if terminated or truncated:
            break
    return {key: torch.stack(values) for key, values in steps.items()}, episode_return


def _agent_rewards(reward, info, n_agents) -> np.ndarray:
    """Preserve an environment reward vector; expand only genuinely shared rewards."""
    if "agent_rewards" in info:
        rewards = np.asarray(info["agent_rewards"], dtype=np.float32)
        if rewards.shape != (n_agents,):
            raise ValueError(f"agent_rewards must have shape {(n_agents,)}, got {rewards.shape}")
        return rewards
    if np.asarray(reward).ndim != 0:
        rewards = np.asarray(reward, dtype=np.float32)
        if rewards.shape != (n_agents,):
            raise ValueError(f"reward vector must have shape {(n_agents,)}, got {rewards.shape}")
        return rewards
    return np.full(n_agents, float(reward), dtype=np.float32)


def _agent_alive(info, n_agents) -> np.ndarray:
    """Preserve released per-agent activity for communication and loss masking."""
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
    gates = torch.zeros_like(actions)
    rewards = torch.zeros(batch_size, horizon, n_agents, device=device)
    mask = torch.zeros_like(rewards)
    alive = torch.zeros_like(rewards)
    continuation = torch.zeros_like(rewards)
    for index, episode in enumerate(episodes):
        length = episode["obs"].shape[0]
        obs[index, :length] = episode["obs"]
        actions[index, :length] = episode["actions"]
        gates[index, :length] = episode["gates"]
        rewards[index, :length] = episode["rewards"]
        mask[index, :length] = 1.0
        alive[index, :length] = episode["alive"]
        continuation[index, :length] = episode["continuation"]
    return {
        "obs": obs,
        "actions": actions,
        "gates": gates,
        "rewards": rewards,
        "mask": mask,
        "alive": alive,
        "continuation": continuation,
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
    returns, successes, distances, gates = [], [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        hidden, cell, previous_gate = agent.init_state(1, n_agents, device)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                action = generator.integers(environment.num_actions, size=n_agents)
                gate = generator.integers(2, size=n_agents)
                previous_gate = torch.as_tensor(gate, device=device).unsqueeze(0)
            else:
                action_tensor, _, gate_tensor, _, _, hidden, cell = agent.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    hidden,
                    cell,
                    previous_gate,
                    deterministic=True,
                )
                action = action_tensor.squeeze(0).cpu().numpy()
                gate = gate_tensor.squeeze(0).cpu().numpy()
                previous_gate = gate_tensor
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_return += float(np.mean(_agent_rewards(reward, info, n_agents)))
            gates.extend(np.asarray(gate).tolist())
            if terminated or truncated:
                break
        returns.append(episode_return)
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {
        "returns": returns,
        "successes": successes,
        "mean_distances": distances,
        "communication_rate": float(np.mean(gates)) if gates else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train faithful IC3Net.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="ic3net.pt")
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
