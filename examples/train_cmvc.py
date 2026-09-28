"""Train CMVC on bounded discrete cooperative navigation.

The library learner owns the recurrent CTDE updates and counterfactual supervision.
This script collects complete episodes because the paper samples recurrent trajectories.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl import CMVCConfig, CMVCLearner
from modmarl.algorithms.cmvc import PAPER_SOURCE


def train(
    *,
    env: str = "paper_navigation",
    n_agents: int = 3,
    horizon: int = 40,
    episodes: int = 5_000,
    seed: int = 7,
    config: CMVCConfig | None = None,
    updates_per_episode: int | None = None,
    evaluation_episodes: int = 64,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    """Run CMVC with paper network/optimizer settings and episode replay."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    n_agents = environment.n_agents
    architecture = config or CMVCConfig()
    learner = CMVCLearner(
        n_agents,
        environment.obs_dim,
        environment.num_actions,
        environment.horizon,
        architecture,
    ).to(dev)
    updates_per_episode = environment.horizon if updates_per_episode is None else updates_per_episode
    if updates_per_episode < 0:
        raise ValueError("updates_per_episode must be nonnegative")
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
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        hidden = learner.policy.initial_hidden(1, dev)
        observations = [obs.copy()]
        actions, rewards, dones = [], [], []
        episode_return = 0.0
        for _ in range(environment.horizon):
            output = learner.policy.step(
                torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0),
                hidden,
                communication_enabled=learner.communication_enabled,
                hard=True,
            )
            hidden = output.hidden.detach()
            action_indices = output.action_indices.squeeze(0).cpu().numpy()
            env_actions = (
                output.action_vectors.squeeze(0).detach().cpu().numpy()
                if env.startswith("paper_")
                else action_indices
            )
            next_obs, reward, terminated, truncated, _ = environment.step(env_actions)
            observations.append(next_obs.copy())
            actions.append(action_indices)
            rewards.append(float(reward))
            dones.append(float(terminated))
            obs = next_obs
            episode_return += float(reward)
            if terminated or truncated:
                break
        learner.replay.add_episode(
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
        for _ in range(updates_per_episode):
            if learner.update() is not None:
                update_count += 1
        returns.append(episode_return)
        if episode % 100 == 0:
            print(f"episode {episode:5d}  return {episode_return:8.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {
                "learner": learner.state_dict(),
                "actor_optimizer": learner.actor_optimizer.state_dict(),
                "critic_optimizer": learner.critic_optimizer.state_dict(),
                "selector_optimizer": learner.selector_optimizer.state_dict(),
                "config": architecture,
                "updates": learner.update_count,
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
        "algorithm": "cmvc",
        "env": env,
        "episodes": episodes,
        "n_agents": n_agents,
        "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "updates": update_count,
        "checkpoint": checkpoint,
        "source_revision": PAPER_SOURCE,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "communication_rate": final_evaluation["communication_rate"],
        "validation_criterion": {
            "return_margin_over_random": 0.0,
            "scope": "every confirmation seed",
        },
        "config": {
            "actor_hidden_dim": architecture.actor_hidden_dim,
            "critic_hidden_dim": architecture.critic_hidden_dim,
            "hyper_hidden_dim": architecture.hyper_hidden_dim,
            "message_hidden_dim": architecture.message_hidden_dim,
            "gamma": architecture.gamma,
            "actor_learning_rate": architecture.actor_learning_rate,
            "critic_learning_rate": architecture.critic_learning_rate,
            "selector_learning_rate": architecture.selector_learning_rate,
            "batch_size": architecture.batch_size,
            "replay_capacity": architecture.replay_capacity,
            "communication_warmup_updates": architecture.communication_warmup_updates,
            "target_update_interval": architecture.target_update_interval,
            "pruning_percentile": architecture.pruning_percentile,
            "updates_per_episode": updates_per_episode,
        },
    }


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
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes, distances, requests = [], [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        hidden = learner.policy.initial_hidden(1, device)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            if random_policy:
                action = generator.integers(environment.num_actions, size=environment.n_agents)
            else:
                output = learner.policy.step(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    hidden,
                    communication_enabled=learner.communication_enabled,
                    deterministic=True,
                )
                hidden = output.hidden
                action = (
                    output.action_vectors.squeeze(0).cpu().numpy()
                    if env_name.startswith("paper_")
                    else output.action_indices.squeeze(0).cpu().numpy()
                )
                off_diagonal = ~torch.eye(environment.n_agents, dtype=torch.bool, device=device)
                requests.extend(output.request_gates.squeeze(0)[off_diagonal].cpu().tolist())
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_return += float(reward)
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
        "communication_rate": float(np.mean(requests)) if requests else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMVC on cooperative navigation.")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="cmvc.pt")
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        n_agents=args.n_agents,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
