"""Train CDC (connectivity-driven communication) on a cooperative MPE task and save a checkpoint.

Run:
    python examples/train_cdc.py --episodes 400

CDC's policy exchanges pairwise messages over a learned graph and diffuses them with a
heat kernel; a centralized LSTM critic scores the joint action. Off-policy actor-critic
with straight-through Gumbel actions. `train()` is importable; built from the public
modMARL API.

The defaults preserve the paper's training path: the actor receives only its aggregated
message (Eq. 7), the archived Navigation Control scenario supplies the observations and
rewards, and the networks update after every 100 new replay samples (Sect. 4.2). The
short CLI run is a demonstration; the paper trained Navigation Control for 100,000
episodes.

``diffusion_steps``, ``diffusion_max`` and ``stable_delta`` configure Equation (5)'s
search grid; the defaults are the paper's P=300 over (0, 15] with s=0.05.

This is the core CDC method. The factorised, multiscale and spectral-operator variants
were developed in a separate research repository -- they were explored after the
published algorithm and are not paper ablations or part of modMARL's public API.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl import CDCAgent, CDCConfig, ReplayBuffer
from modmarl.common.replay import ReplayBatch


def train(
    *,
    env: str = "paper_navigation",
    n_agents: int = 3,
    horizon: int = 50,
    episodes: int = 400,
    seed: int = 7,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    hidden_dim: int = 64,
    message_dim: int = 64,
    diffusion_steps: int = 300,
    diffusion_max: float = 15.0,   # paper Sect. 4.2 grid upper bound
    stable_delta: float = 0.05,
    buffer_size: int = 1_000_000,
    batch_size: int = 1024,
    warmup_steps: int = 1024,
    update_every: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
    variant: str = "clean",
    evaluation_episodes: int = 100,
    evaluation_seed: int = 20_001,
) -> dict:
    if update_every < 1:
        raise ValueError("update_every must be at least 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    if variant not in {"clean", "released"}:
        raise ValueError("variant must be 'clean' or 'released'")
    agent = CDCAgent(
        obs_dim=environment.obs_dim,
        action_dim=environment.num_actions,
        hidden_dim=hidden_dim,
        message_dim=message_dim,
        variant="paper" if variant == "clean" else "released",
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        diffusion_steps=diffusion_steps,
        diffusion_max=diffusion_max,
        stable_delta=stable_delta,
    )
    schedule = CDCConfig(batch_size=batch_size, update_interval=update_every)
    replay = ReplayBuffer(buffer_size, environment.n_agents, environment.obs_dim)
    initial_evaluation = _evaluate(
        agent, env, environment.n_agents, horizon, evaluation_seed, evaluation_episodes,
    )
    random_evaluation = _evaluate(
        agent,
        env,
        environment.n_agents,
        horizon,
        evaluation_seed,
        evaluation_episodes,
        random_policy=True,
    )

    total_steps = 0
    update_count = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset()
        episode_return = 0.0
        for _ in range(environment.horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_idx = agent.act(obs_t)
            action = action_idx.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = environment.step(action)
            replay.add(obs=obs, actions=action, reward=reward, next_obs=next_obs, done=terminated)
            obs = next_obs
            episode_return += reward
            total_steps += 1

            if (
                total_steps >= warmup_steps
                and schedule.update_due(environment_steps=total_steps, replay_size=len(replay))
            ):
                agent.to(dev)
                batch = _sample_without_replacement(replay, batch_size, dev)
                reward_values = replay.rewards[: replay.size]
                reward_std = max(float(reward_values.std()), 1e-8)
                batch = ReplayBatch(
                    obs=batch.obs,
                    actions=batch.actions,
                    rewards=(batch.rewards - float(reward_values.mean())) / reward_std,
                    next_obs=batch.next_obs,
                    dones=batch.dones,
                )
                agent.update(batch.obs, batch.actions, batch.rewards, batch.next_obs, batch.dones)
                agent.cpu()
                update_count += 1

            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "agent": agent.state_dict(),
                "actor_optimizer": agent.actor_optimizer.state_dict(),
                "critic_optimizer": agent.critic_optimizer.state_dict(),
                "environment_steps": total_steps,
                "updates": update_count,
                "variant": agent.variant,
            },
            checkpoint,
        )

    final_evaluation = _evaluate(
        agent, env, environment.n_agents, horizon, evaluation_seed, evaluation_episodes,
    )
    ablated_evaluation = _evaluate(
        agent, env, environment.n_agents, horizon, evaluation_seed, evaluation_episodes,
        ablate_messages=True,
    )

    return {
        "algorithm": "cdc",
        "variant": variant,
        "env": env,
        "episodes": episodes,
        "n_agents": environment.n_agents,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "updates": update_count,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "message_ablated_evaluation": ablated_evaluation,
        "validation_criterion": {
            "return_margin_over_initial": 3.0,
            "return_margin_over_no_message": 1.0,
            "scope": "mean over held-out episodes for every confirmation seed",
        },
        "communication_rate": 1.0,
        "config": {
            "env": env,
            "episodes": episodes,
            "seed": seed,
            "n_agents": environment.n_agents,
            "horizon": environment.horizon,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "hidden_dim": hidden_dim,
            "message_dim": message_dim,
            "diffusion_steps": diffusion_steps,
            "diffusion_max": diffusion_max,
            "stable_delta": stable_delta,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "update_every": update_every,
            "variant": variant,
        },
        "checkpoint": checkpoint,
    }


def _sample_without_replacement(replay: ReplayBuffer, batch_size: int, device: torch.device) -> ReplayBatch:
    """Sample CDC replay exactly as the released buffer: unique indices per update."""
    indices = np.random.choice(replay.size, size=batch_size, replace=False)
    return ReplayBatch(
        obs=torch.as_tensor(replay.obs[indices], device=device),
        actions=torch.as_tensor(replay.actions[indices], device=device),
        rewards=torch.as_tensor(replay.rewards[indices], device=device),
        next_obs=torch.as_tensor(replay.next_obs[indices], device=device),
        dones=torch.as_tensor(replay.dones[indices], device=device),
    )


def _evaluate(
    agent: CDCAgent,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    *,
    ablate_messages: bool = False,
    random_policy: bool = False,
) -> dict[str, list[float]]:
    """Evaluate deterministic actions with either learned or zeroed communication."""
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    returns, distances, successes = [], [], []
    generator = np.random.default_rng(seed)
    try:
        environment = make_env(env_name, n_agents, horizon, seed)
        for episode in range(episodes):
            obs, _ = environment.reset(seed=seed + episode)
            episode_return = 0.0
            info = {}
            for _ in range(environment.horizon):
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                if random_policy:
                    actions = torch.as_tensor(
                        generator.integers(0, environment.num_actions, size=environment.n_agents),
                    ).unsqueeze(0)
                elif ablate_messages:
                    actions = agent.act_without_messages(obs_t)
                else:
                    actions = agent.act(obs_t, deterministic=True)
                obs, reward, terminated, truncated, info = environment.step(actions.squeeze(0).numpy())
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
            distances.append(float(info.get("mean_distance", float("nan"))))
            successes.append(float(bool(info.get("success", False))))
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    return {"returns": returns, "mean_distances": distances, "successes": successes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CDC on a cooperative MPE task.")
    parser.add_argument("--env", default="navigation")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="cdc.pt")
    parser.add_argument("--variant", choices=("clean", "released"), default="clean")
    args = parser.parse_args()
    train(
        env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed,
        checkpoint=args.checkpoint, variant=args.variant,
    )


if __name__ == "__main__":
    main()
