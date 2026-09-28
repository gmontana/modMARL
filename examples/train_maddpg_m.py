"""Train MADDPG-M on the noisy-observation navigation task and save a checkpoint.

Run:
    python examples/train_maddpg_m.py --episodes 400

Two-level training: every C steps each agent emits a broadcast *willingness* (communication
policy nu); an argmax selects whose observation becomes the shared medium, which remains
fixed for the next C action steps. Every agent acts on its own observation plus that medium.
Separate action- and communication-level replay streams preserve the two time scales. Action policies are
trained on the environment's intrinsic reward (reaching the landmarks encoded in the medium)
and communication policies on the accumulated extrinsic reward — so the team learns to broadcast
the gifted agent's (true) observation. `train()` is importable; built from the public API and
the NoisyNavigationEnv showcase environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl_envs.noisy_navigation import NoisyNavigationEnv
from modmarl import MADDPGMAgent
from modmarl.common.replay import MADDPGMReplayBuffer

GAMMA = 0.85
TAU = 0.01
POLICY_REG = 1e-3
GRAD_CLIP = 0.5
OU_THETA = 0.15
OU_SIGMA = 0.2


class OUNoise:
    """Ornstein–Uhlenbeck exploration used by the paper for both policy levels."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.state = np.zeros(shape, dtype=np.float32)

    def sample(self) -> np.ndarray:
        self.state += OU_THETA * -self.state + OU_SIGMA * np.random.standard_normal(self.state.shape)
        return self.state.astype(np.float32)


def train(
    *,
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 400,
    seed: int = 7,
    noise: float = 1.0,
    gifted_agent: int = 0,
    learning_rate: float = 1e-2,
    hidden_dim: int = 64,
    critic_hidden_dim: int = 128,
    communication_interval: int = 5,
    buffer_size: int = 1_000_000,
    batch_size: int = 1024,
    warmup_steps: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
    evaluation_episodes: int = 100,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    env = NoisyNavigationEnv(n_agents=n_agents, horizon=horizon, noise=noise, gifted_agent=gifted_agent, seed=seed)
    n, obs_dim, num_actions = env.n_agents, env.obs_dim, env.num_actions
    agents = [
        MADDPGMAgent(n, obs_dim, num_actions, hidden_dim, critic_hidden_dim).to(dev)
        for _ in range(n)
    ]

    comm_policy_opts = [torch.optim.Adam(a.comm_policy.parameters(), lr=learning_rate) for a in agents]
    action_policy_opts = [torch.optim.Adam(a.action_policy.parameters(), lr=learning_rate) for a in agents]
    comm_critic_opts = [torch.optim.Adam(a.comm_critic.parameters(), lr=learning_rate) for a in agents]
    action_critic_opts = [torch.optim.Adam(a.action_critic.parameters(), lr=learning_rate) for a in agents]
    action_replay = MADDPGMReplayBuffer(buffer_size, n, obs_dim, num_actions)
    comm_replay = MADDPGMReplayBuffer(buffer_size, n, obs_dim, num_actions)

    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agents, n_agents=n, horizon=horizon, noise=noise,
        gifted_agent=gifted_agent, communication_interval=communication_interval,
        seed=evaluation_seed, episodes=evaluation_episodes, device=dev,
    )
    random_evaluation = _evaluate(
        agents, n_agents=n, horizon=horizon, noise=noise,
        gifted_agent=gifted_agent, communication_interval=communication_interval,
        seed=evaluation_seed, episodes=evaluation_episodes, device=dev,
        random_policy=True,
    )

    total_steps = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        comm_noise = OUNoise((n,))
        action_noise = OUNoise((n, num_actions))
        comm_obs = obs.copy()
        comm = None
        medium = None
        speaker = 0
        accumulated_reward = 0.0
        for step in range(env.horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            with torch.no_grad():
                if step % communication_interval == 0:
                    comm_obs = obs.copy()
                    comm = torch.cat([agents[i].comm_policy(obs_t[i:i + 1]) for i in range(n)]).squeeze(-1)
                    comm = (comm + torch.as_tensor(comm_noise.sample(), device=dev)).clamp(0.0, 1.0)
                    speaker = int(torch.argmax(comm).item())
                    medium = obs[speaker].copy()
                    accumulated_reward = 0.0
                medium_t = torch.as_tensor(medium, dtype=torch.float32, device=dev).unsqueeze(0)
                if total_steps < warmup_steps:
                    action = np.random.uniform(0.0, 1.0, size=(n, num_actions)).astype(np.float32)
                else:
                    action = np.concatenate(
                        [agents[i].action_policy(obs_t[i:i + 1], medium_t).cpu().numpy() for i in range(n)], axis=0,
                    )
                    action = np.clip(action + action_noise.sample(), 0.0, 1.0)

            next_obs, ext_reward, terminated, truncated, _ = env.step(action)
            action_replay.add(
                obs=obs, comm_actions=comm.cpu().numpy(), medium=medium, actions=action,
                ext_reward=ext_reward, int_reward=env.intrinsic_reward(speaker), next_obs=next_obs, done=terminated,
            )
            accumulated_reward += ext_reward
            chunk_done = (step + 1) % communication_interval == 0 or terminated or truncated
            if chunk_done:
                comm_replay.add(
                    obs=comm_obs, comm_actions=comm.cpu().numpy(), medium=medium, actions=action,
                    ext_reward=accumulated_reward, int_reward=0.0, next_obs=next_obs,
                    done=terminated,
                )
            obs = next_obs
            episode_return += ext_reward
            total_steps += 1

            if len(action_replay) >= batch_size and len(comm_replay) >= batch_size and total_steps % 100 == 0:
                _update(agents, comm_policy_opts, action_policy_opts, comm_critic_opts, action_critic_opts,
                        action_replay.sample(batch_size, dev), comm_replay.sample(batch_size, dev), num_actions)

            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    env.close()
    if checkpoint is not None:
        torch.save([a.state_dict() for a in agents], checkpoint)

    final_evaluation = _evaluate(
        agents, n_agents=n, horizon=horizon, noise=noise,
        gifted_agent=gifted_agent, communication_interval=communication_interval,
        seed=evaluation_seed, episodes=evaluation_episodes, device=dev,
    )
    return {
        "algorithm": "maddpg_m",
        "env": "noisy_navigation",
        "episodes": episodes,
        "n_agents": n,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "validation_criterion": {
            "return_improvement_fraction_of_abs_random": 0.2,
            "scope": "every confirmation seed",
        },
        "communication_rate": 1.0 / communication_interval,
        "config": {
            "episodes": episodes,
            "seed": seed,
            "n_agents": n,
            "horizon": horizon,
            "noise": noise,
            "gifted_agent": gifted_agent,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "critic_hidden_dim": critic_hidden_dim,
            "communication_interval": communication_interval,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "gamma": GAMMA,
            "tau": TAU,
        },
        "checkpoint": checkpoint,
    }


@torch.no_grad()
def _target_medium(agents, next_obs: torch.Tensor) -> torch.Tensor:
    """Select the next-step broadcast medium from the target communication policies."""
    comm = torch.stack(
        [agent.target_comm_policy(next_obs[:, agent_id]).squeeze(-1) for agent_id, agent in enumerate(agents)],
        dim=1,
    )
    speaker = comm.argmax(dim=1)
    batch_index = torch.arange(next_obs.shape[0], device=next_obs.device)
    return next_obs[batch_index, speaker]


def _update(agents, comm_policy_opts, action_policy_opts, comm_critic_opts, action_critic_opts,
            action_batch, comm_batch, num_actions):
    n = len(agents)
    del num_actions

    for i, agent in enumerate(agents):
        # --- Action level (mu): trained on the intrinsic reward and the next-step target medium. ---
        with torch.no_grad():
            next_action = agent.target_action_policy(action_batch.next_obs[:, i], action_batch.medium)
            y_mu = action_batch.int_rewards + GAMMA * (1.0 - action_batch.dones) * agent.target_action_critic(
                action_batch.next_obs[:, i], action_batch.medium, next_action,
            )
        action_critic_loss = F.mse_loss(
            agent.action_critic(action_batch.obs[:, i], action_batch.medium, action_batch.actions[:, i]), y_mu,
        )
        action_critic_opts[i].zero_grad(set_to_none=True)
        action_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.action_critic.parameters(), GRAD_CLIP)
        action_critic_opts[i].step()

        # --- Communication level (nu): trained on the extrinsic reward, centralized critic. ---
        with torch.no_grad():
            next_comm = torch.stack([agents[j].target_comm_policy(comm_batch.next_obs[:, j]).squeeze(-1) for j in range(n)], dim=1)
            y_nu = comm_batch.ext_rewards + GAMMA * (1.0 - comm_batch.dones) * agent.target_comm_critic(comm_batch.next_obs, next_comm)
        comm_critic_loss = F.mse_loss(agent.comm_critic(comm_batch.obs, comm_batch.comm_actions), y_nu)
        comm_critic_opts[i].zero_grad(set_to_none=True)
        comm_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.comm_critic.parameters(), GRAD_CLIP)
        comm_critic_opts[i].step()

    for i, agent in enumerate(agents):
        # Action policy: maximise the action critic's value of its own (medium-informed) action.
        action = agent.action_policy(action_batch.obs[:, i], action_batch.medium)
        action_policy_loss = -agent.action_critic(action_batch.obs[:, i], action_batch.medium, action).mean()
        action_policy_loss = action_policy_loss + POLICY_REG * action.pow(2).mean()
        action_policy_opts[i].zero_grad(set_to_none=True)
        action_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.action_policy.parameters(), GRAD_CLIP)
        action_policy_opts[i].step()

        # Communication policy: maximise the comm critic's value of its own broadcast willingness.
        comm_all = comm_batch.comm_actions.clone()
        comm_all[:, i] = agent.comm_policy(comm_batch.obs[:, i]).squeeze(-1)
        comm_policy_loss = -agent.comm_critic(comm_batch.obs, comm_all).mean()
        comm_policy_opts[i].zero_grad(set_to_none=True)
        comm_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.comm_policy.parameters(), GRAD_CLIP)
        comm_policy_opts[i].step()

    for agent in agents:
        agent.soft_update(TAU)


@torch.no_grad()
def _evaluate(
    agents, *, n_agents, horizon, noise, gifted_agent, communication_interval,
    seed, episodes, device, random_policy=False,
):
    env = NoisyNavigationEnv(
        n_agents=n_agents, horizon=horizon, noise=noise, gifted_agent=gifted_agent, seed=seed,
    )
    generator = np.random.default_rng(seed)
    returns, successes, distances, correct, decisions = [], [], [], 0, 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return, info = 0.0, {}
        medium = None
        for step in range(horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if step % communication_interval == 0:
                speaker = (
                    int(generator.integers(n_agents))
                    if random_policy
                    else int(torch.cat([
                        agent.comm_policy(obs_t[i:i + 1])
                        for i, agent in enumerate(agents)
                    ]).squeeze(-1).argmax().item())
                )
                correct += int(speaker == gifted_agent)
                decisions += 1
                medium = obs_t[speaker:speaker + 1]
            action = (
                generator.uniform(0.0, 1.0, size=(n_agents, env.num_actions))
                if random_policy
                else np.concatenate([
                    agent.action_policy(obs_t[i:i + 1], medium).cpu().numpy()
                    for i, agent in enumerate(agents)
                ], axis=0)
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    env.close()
    return {
        "returns": [float(value) for value in returns],
        "mean_return": float(np.mean(returns)),
        "successes": successes,
        "mean_distances": distances,
        "communication_accuracy": correct / max(1, decisions),
        "communication_rate": 1.0 / communication_interval,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MADDPG-M on the noisy-observation navigation task.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="maddpg_m.pt")
    args = parser.parse_args()
    train(episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
