"""Train TarMAC with the paper's synchronous centralized actor-critic.

Run:
    python examples/train_tarmac.py --episodes 300

Sixteen complete episodes form an on-policy batch. The recurrent policy is
replayed once to compute categorical log-probabilities and the centralized TD
critic; there is no transition replay, target network, or Gumbel action path.
The fixed horizon is terminal under the paper's finite-horizon objective, so its
last target does not bootstrap.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.tarmac import TarMACAgent, TarMACConfig


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    config: TarMACConfig | None = None,
    batch_episodes: int = 16,
    evaluation_episodes: int = 32,
    evaluation_seed: int | None = None,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    n_agents = environment.n_agents
    agent = TarMACAgent(
        n_agents,
        environment.obs_dim,
        environment.num_actions,
        config,
    ).to(dev)
    evaluation_seed = seed + 100_000 if evaluation_seed is None else evaluation_seed
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
        rollout = []
        for _ in range(min(batch_episodes, episodes - episode_count)):
            episode, episode_return = _collect_episode(
                agent, environment, n_agents, seed + episode_count, dev,
            )
            rollout.append(episode)
            returns.append(episode_return)
            episode_count += 1
        agent.update(**_pad_episodes(rollout, n_agents, environment.obs_dim, dev))
        if episode_count % 32 < len(rollout):
            print(f"episodes {episode_count:4d}  return {returns[-1]:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "agent": agent.state_dict(), "optimizer": agent.optimizer.state_dict(),
                "metadata": {
                    "schema_version": 1, "algorithm": "tarmac", "env": env,
                    "n_agents": n_agents, "horizon": environment.horizon,
                    "constructor": {"n_agents": n_agents, "obs_dim": environment.obs_dim,
                                    "action_dim": environment.num_actions, "config": asdict(agent.config)},
                },
            }, checkpoint,
        )
    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    cfg = agent.config
    validation_criterion = (
        {"final_success_rate": 0.8, "scope": "every confirmation seed"}
        if env == "target_signaling"
        else {
            "return_improvement_fraction_of_abs_random": 0.2,
            "scope": "every confirmation seed",
        }
    )
    return {
        "algorithm": "tarmac",
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
        "validation_criterion": validation_criterion,
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "message_dim": cfg.message_dim,
            "signature_dim": cfg.signature_dim,
            "communication_rounds": cfg.communication_rounds,
            "learning_rate": cfg.learning_rate,
            "rmsprop_alpha": cfg.rmsprop_alpha,
            "gamma": cfg.gamma,
            "entropy_coefficient": cfg.entropy_coefficient,
            "batch_episodes": batch_episodes,
            "optimizer": "RMSprop",
            "learner": "synchronous_centralized_actor_critic",
        },
    }


@torch.no_grad()
def _collect_episode(agent, environment, n_agents, episode_seed, device):
    obs, _ = environment.reset(seed=episode_seed)
    state = agent.policy.initial_state(1, n_agents, device)
    steps = {key: [] for key in ("obs", "actions", "rewards", "continuation")}
    episode_return = 0.0
    for _ in range(environment.horizon):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action, _, _, state, _ = agent.policy.act(obs_tensor.unsqueeze(0), state)
        next_obs, reward, terminated, truncated, _ = environment.step(
            action.squeeze(0).cpu().numpy(),
        )
        team_reward = _team_reward(reward)
        steps["obs"].append(obs_tensor)
        steps["actions"].append(action.squeeze(0))
        steps["rewards"].append(torch.tensor(team_reward, device=device))
        steps["continuation"].append(
            torch.tensor(float(not (terminated or truncated)), device=device),
        )
        obs = next_obs
        episode_return += team_reward
        if terminated or truncated:
            break
    return {key: torch.stack(values) for key, values in steps.items()}, episode_return


def _team_reward(reward) -> float:
    value = np.asarray(reward)
    if value.ndim != 0:
        raise ValueError("TarMAC's paper objective requires one global team reward")
    return float(value)


def _pad_episodes(episodes, n_agents, obs_dim, device):
    batch_size = len(episodes)
    horizon = max(episode["obs"].shape[0] for episode in episodes)
    obs = torch.zeros(batch_size, horizon, n_agents, obs_dim, device=device)
    actions = torch.zeros(batch_size, horizon, n_agents, dtype=torch.long, device=device)
    rewards = torch.zeros(batch_size, horizon, device=device)
    mask = torch.zeros_like(rewards)
    continuation = torch.zeros_like(rewards)
    for index, episode in enumerate(episodes):
        length = episode["obs"].shape[0]
        obs[index, :length] = episode["obs"]
        actions[index, :length] = episode["actions"]
        rewards[index, :length] = episode["rewards"]
        mask[index, :length] = 1.0
        continuation[index, :length] = episode["continuation"]
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "mask": mask,
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
    returns, successes, distances = [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        state = agent.policy.initial_state(1, n_agents, device)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                action = generator.integers(environment.num_actions, size=n_agents)
            else:
                action_tensor, _, _, state, _ = agent.policy.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    state,
                    deterministic=True,
                )
                action = action_tensor.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_return += _team_reward(reward)
            if terminated or truncated:
                break
        returns.append(episode_return)
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    return {
        "returns": returns,
        "successes": successes,
        "mean_distances": distances,
        "communication_rate": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train faithful TarMAC.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="tarmac.pt")
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
