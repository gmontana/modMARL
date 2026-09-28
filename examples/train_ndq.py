"""Train NDQ on a cooperative task and save a checkpoint.

Run:
    python examples/train_ndq.py --episodes 500

QMIX's episode-replay recurrent sibling with a learned pairwise message channel:
whole episodes are collected with the GRU hidden state threaded through time and
replayed in batches.  ``NDQAgent`` owns the complete recurrent double-Q/QMIX
learner, paper losses, optimizer, and targets; this example only collects episodes,
schedules updates, evaluates, and serializes experiment state.

The defaults reproduce the released two-agent Hallway recipe: corridor lengths
[6, 6], message width 3, gamma 0.99, lambda 0.1, beta 0.01, agent identifiers,
no previous-action input, 16 parallel-rollout-equivalent episodes per learner
update, and epsilon annealing over 50,000 environment steps.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.ndq import NDQAgent
from modmarl.common.replay import EpisodeReplayBuffer

GRAD_CLIP = 10.0


def train(
    *,
    env: str = "ndq_hallway",
    n_agents: int = 2,
    horizon: int = 16,
    episodes: int = 100_000,
    seed: int = 7,
    lr: float = 5e-4,
    hidden_dim: int = 64,
    message_dim: int = 3,
    mixer_hidden_dim: int = 32,
    buffer_episodes: int = 5000,
    batch_episodes: int = 32,
    warmup_episodes: int = 32,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_anneal_steps: int | None = 50_000,
    gamma: float = 0.99,
    c_beta: float = 0.1,
    comm_beta: float = 1e-2,
    updates_per_episode: int = 1,
    update_every_episodes: int = 16,
    target_update_every: int = 200,
    evaluation_episodes: int = 300,
    include_agent_id: bool = True,
    include_last_action: bool = False,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    n, obs_dim, num_actions = environment.n_agents, environment.obs_dim, environment.num_actions
    agent = NDQAgent(
        n_agents=n, obs_dim=obs_dim, action_dim=num_actions,
        hidden_dim=hidden_dim, message_dim=message_dim, mixer_hidden_dim=mixer_hidden_dim,
        learning_rate=lr, gamma=gamma, communication_weight=c_beta,
        succinctness_weight=comm_beta, max_grad_norm=GRAD_CLIP,
        include_agent_id=include_agent_id, include_last_action=include_last_action,
    ).to(dev)
    replay = EpisodeReplayBuffer(buffer_episodes, environment.horizon, n, obs_dim)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        random_policy=True,
    )

    returns: list[float] = []
    total_steps = 0
    for episode in range(episodes):
        schedule_position = total_steps if epsilon_anneal_steps is not None else episode
        schedule_length = epsilon_anneal_steps if epsilon_anneal_steps is not None else max(1, episodes - 1)
        frac = min(1.0, schedule_position / max(1, schedule_length))
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        episode_return, episode_steps = _collect_episode(
            agent, environment, replay, epsilon, num_actions, seed + episode, dev,
        )
        total_steps += episode_steps
        returns.append(episode_return)

        if (
            len(replay) >= max(batch_episodes, warmup_episodes)
            and (episode + 1) % update_every_episodes == 0
        ):
            for _ in range(updates_per_episode):
                agent.update(replay.sample(batch_episodes, dev))
        agent.update_targets_if_due(episode, target_update_every)
        if episode % 20 == 0:
            print(f"episode {episode:4d}  eps {epsilon:4.2f}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save({"model": agent.state_dict(), "optimizer": agent.optimizer.state_dict()}, checkpoint)

    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    ablated_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        ablate_messages=True,
    )
    if env == "ndq_hallway":
        validation_criterion = {
            "return_margin_over_random": 8.0,
            "minimum_success_rate": 0.90,
            "scope": "every confirmation seed",
        }
    else:
        validation_criterion = {
            "return_margin_over_random": 3.0,
            "maximum_mean_distance": 0.80,
            "scope": "every confirmation seed",
        }
    return {
        "algorithm": "ndq",
        "env": env,
        "episodes": episodes,
        "n_agents": n,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "communication_rate": 1.0,
        "message_ablated_evaluation": ablated_evaluation,
        "validation_criterion": validation_criterion,
    }


@torch.no_grad()
def _collect_episode(agent, environment, replay, epsilon, num_actions, episode_seed, dev) -> tuple[float, int]:
    obs, _ = environment.reset(seed=episode_seed)
    hidden = agent.init_hidden(1, dev)
    n = environment.n_agents
    obs_list, action_list, reward_list, done_list = [obs], [], [], []
    last_actions = None
    episode_return = 0.0
    for _ in range(environment.horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        m_in, _ = agent.messages(obs_t, last_actions)
        q, hidden = agent.q_step(obs_t, m_in, hidden, last_actions)
        greedy = q.argmax(dim=-1).squeeze(0).cpu().numpy()
        explore = np.random.random(n) < epsilon
        action = np.where(explore, np.random.randint(0, num_actions, size=n), greedy).astype(np.int64)
        last_actions = torch.as_tensor(
            np.eye(num_actions, dtype=np.float32)[action], device=dev,
        ).unsqueeze(0)

        next_obs, reward, terminated, truncated, _ = environment.step(action)
        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(float(terminated))
        obs = next_obs
        episode_return += reward
        if terminated or truncated:
            break

    replay.add_episode(
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(action_list, dtype=np.int64),
        rewards=np.asarray(reward_list, dtype=np.float32),
        dones=np.asarray(done_list, dtype=np.float32),
    )
    return episode_return, len(action_list)


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
    random_policy=False,
    ablate_messages=False,
):
    """Evaluate held-out returns, navigation metrics, and the message ablation."""
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    returns, successes, distances = [], [], []
    try:
        for episode in range(episodes):
            obs, _ = environment.reset(seed=seed + episode)
            hidden = agent.init_hidden(1, device)
            last_actions = None
            episode_return, info = 0.0, {}
            for _ in range(environment.horizon):
                if random_policy:
                    action = generator.integers(environment.num_actions, size=environment.n_agents)
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    threshold = float("inf") if ablate_messages else None
                    messages, _ = agent.messages(
                        obs_tensor, last_actions, drop_threshold=threshold,
                    )
                    q_values, hidden = agent.q_step(
                        obs_tensor, messages, hidden, last_actions,
                    )
                    action = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()
                    last_actions = torch.as_tensor(
                        np.eye(environment.num_actions, dtype=np.float32)[action], device=device,
                    ).unsqueeze(0)
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            successes.append(float(bool(info.get("success", False))))
            distances.append(float(info.get("mean_distance", float("nan"))))
    finally:
        torch.random.set_rng_state(rng_state)
    return {"returns": returns, "successes": successes, "mean_distances": distances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NDQ on a cooperative task.")
    parser.add_argument("--env", default="ndq_hallway")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="ndq.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
