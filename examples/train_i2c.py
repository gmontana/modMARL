"""Train I2C on a cooperative MPE task and save a checkpoint.

Run:
    python examples/train_i2c.py --episodes 300

A parameter-shared prior-gated decentralized policy and centralized critic are trained
off-policy (MADDPG-style) with the release's soft Gumbel actions. A full-communication
CTDE teacher first supplies percentile-thresholded causal-influence labels. The fitted
prior is then frozen in a fresh learner whose requested raw observations pass through
the paper's two-layer recurrent encoder. Communication is one-shot per step. `train()`
is importable and built from the public modMARL API and the I2C module.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import PaperParticleEnv, make_env
from modmarl.algorithms.i2c import I2CAgent, I2CReplayBuffer

GAMMA = 0.95
TAU = 0.01


def train(
    *, env: str = "simple_spread", n_agents: int = 3, horizon: int = 25,
    episodes: int = 300, seed: int = 7, policy_lr: float = 1e-2,
    critic_lr: float = 1e-2, prior_lr: float = 1e-2, hidden_dim: int = 128,
    message_dim: int | None = None, threshold: float = 0.5,
    buffer_size: int = 1_000_000, batch_size: int = 800,
    warmup_steps: int = 32_000, update_interval: int = 100,
    device: str = "cpu", checkpoint: str | None = None,
    evaluation_episodes: int = 32, teacher_episodes: int | None = None,
    causal_samples: int = 4096, prior_steps: int = 1000,
    prior_batch_size: int = 512,
) -> dict:
    """Run the paper's teacher-prior-frozen-policy training protocol."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    n = environment.n_agents

    def make_agent() -> I2CAgent:
        return I2CAgent(
            n_agents=n, obs_dim=environment.obs_dim, action_dim=environment.num_actions,
            message_dim=message_dim, hidden_dim=hidden_dim, threshold=threshold,
            policy_lr=policy_lr, critic_lr=critic_lr, prior_lr=prior_lr,
        ).to(dev)

    teacher = make_agent()
    teacher_count = episodes if teacher_episodes is None else teacher_episodes
    teacher_returns, teacher_replay = _train_phase(
        teacher, environment, teacher_count, seed, dev, buffer_size, batch_size,
        warmup_steps, update_interval, full_communication=True,
    )
    prior_obs, prior_locations, influences = _causal_dataset(
        teacher, teacher_replay, causal_samples, dev,
    )
    influence_threshold = torch.quantile(influences, teacher.influence_percentile / 100.0)
    prior_labels = influences >= influence_threshold
    prior_loss = teacher.fit_prior(
        prior_obs, prior_locations, prior_labels, steps=prior_steps,
        batch_size=prior_batch_size,
    )

    torch.manual_seed(seed + 1_000_000)
    agent = make_agent()
    agent.load_frozen_prior(teacher)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        random_policy=True,
    )
    returns, _ = _train_phase(
        agent, environment, episodes, seed + 1_000_000, dev, buffer_size,
        batch_size, warmup_steps, update_interval, full_communication=False,
    )

    if checkpoint is not None:
        torch.save({
            "model": agent.state_dict(),
            "policy_optimizer": agent.policy_optimizer.state_dict(),
            "critic_optimizer": agent.critic_optimizer.state_dict(),
            "prior_state": agent.policy.prior_net.state_dict(),
            "influence_threshold": float(influence_threshold),
        }, checkpoint)
    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    message_ablated_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        no_communication=True,
    )
    return {
        "algorithm": "i2c", "env": env, "episodes": episodes,
        "teacher_episodes": teacher_count, "n_agents": n,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0, "returns": returns,
        "teacher_returns": teacher_returns, "prior_examples": int(prior_labels.numel()),
        "prior_positive_rate": float(prior_labels.float().mean()),
        "prior_loss": prior_loss, "influence_threshold": float(influence_threshold),
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation, "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "communication_rate": float(np.mean(final_evaluation["communication_rates"])),
        "message_ablated_evaluation": message_ablated_evaluation,
        "validation_criterion": {
            "return_margin_over_random": 2.0, "maximum_mean_distance": 0.85,
            "scope": "every confirmation seed",
        },
    }


def _train_phase(
    agent, environment, episodes, seed, device, buffer_size, batch_size,
    warmup_steps, update_interval, *, full_communication,
):
    replay = I2CReplayBuffer(
        buffer_size, environment.n_agents, environment.obs_dim, environment.num_actions,
    )
    total_steps = 0
    returns = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        locations, mask = _communication_candidates(environment)
        episode_return = 0.0
        for _ in range(environment.horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            locations_t = torch.as_tensor(locations, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, device=device).unsqueeze(0)
            with torch.no_grad():
                gate = mask_t.to(obs_t.dtype) if full_communication else agent.policy._gate(
                    agent.policy.prior(obs_t, locations_t), mask_t,
                )
                messages_t = agent.policy.pack_messages(obs_t, gate)
                action_vector, action_idx, _ = agent.policy.sample_from_messages(obs_t, messages_t)
            replay_action = action_vector.squeeze(0).cpu().numpy()
            env_action = replay_action if isinstance(environment, PaperParticleEnv) else (
                action_idx.squeeze(0).cpu().numpy()
            )
            next_obs, reward, terminated, truncated, _ = environment.step(env_action)
            next_locations, next_mask = _communication_candidates(environment)
            replay.add(
                obs, replay_action, reward, next_obs, terminated,
                candidate_locations=locations, candidate_mask=mask,
                next_candidate_locations=next_locations, next_candidate_mask=next_mask,
                messages=messages_t.squeeze(0).cpu().numpy(),
            )
            obs, locations, mask = next_obs, next_locations, next_mask
            episode_return += reward
            total_steps += 1
            if (
                len(replay) >= batch_size and total_steps >= warmup_steps
                and total_steps % update_interval == 0
            ):
                agent.update(
                    replay.sample(batch_size, device), gamma=GAMMA, tau=TAU,
                    full_communication=full_communication,
                )
            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 100 == 0:
            print(f"episode {episode:5d}  return {episode_return:8.2f}", flush=True)
    return returns, replay


@torch.no_grad()
def _causal_dataset(agent, replay, sample_count, device):
    batch = replay.sample(min(sample_count, len(replay)), device)
    mask = batch.candidate_mask.bool()
    observations, locations, values = [], [], []
    for receiver in range(agent.policy.n_agents):
        selected = torch.arange(batch.obs.shape[0], device=device) % agent.policy.n_agents == receiver
        if not selected.any():
            continue
        influence = agent.causal_influence(
            batch.obs[selected], batch.actions[selected], receiver=receiver,
            candidate_mask=mask[selected],
        )[:, receiver]
        valid = mask[selected, receiver]
        own_obs = batch.obs[selected, receiver].unsqueeze(1).expand(-1, valid.shape[1], -1)
        observations.append(own_obs[valid])
        locations.append(batch.candidate_locations[selected, receiver][valid])
        values.append(influence[valid])
    return torch.cat(observations), torch.cat(locations), torch.cat(values)


@torch.no_grad()
def _evaluate(
    agent, env_name, n_agents, horizon, seed, episodes, device, *, random_policy=False,
    no_communication=False,
):
    """Matched deterministic evaluation and random-policy reference."""
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes, distances, communication_rates = [], [], [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        candidate_locations, candidate_mask = _communication_candidates(environment)
        episode_return, info, episode_rates = 0.0, {}, []
        for _ in range(environment.horizon):
            if random_policy:
                action = generator.integers(environment.num_actions, size=environment.n_agents)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                locations = torch.as_tensor(candidate_locations, dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.as_tensor(candidate_mask, device=device).unsqueeze(0)
                if no_communication:
                    mask = torch.zeros_like(mask)
                prior_logits = agent.policy.prior(obs_tensor, locations)
                valid_count = mask.sum().clamp_min(1)
                gate = agent.policy._gate(prior_logits, mask)
                episode_rates.append(float(gate.sum() / valid_count))
                messages = agent.policy.pack_messages(obs_tensor, gate)
                action = agent.policy.sample_from_messages(
                    obs_tensor, messages, deterministic=True,
                )[1].squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = environment.step(action)
            candidate_locations, candidate_mask = _communication_candidates(environment)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
        communication_rates.append(float(np.mean(episode_rates)) if episode_rates else 0.0)
    return {
        "returns": returns, "successes": successes, "mean_distances": distances,
        "communication_rates": communication_rates,
    }


def _communication_candidates(environment):
    if hasattr(environment, "communication_candidates"):
        return environment.communication_candidates()
    n = environment.n_agents
    return np.zeros((n, n, 2), dtype=np.float32), ~np.eye(n, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train I2C on a cooperative MPE task.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="i2c.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
