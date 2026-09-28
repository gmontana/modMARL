"""Train ExpoComm on a cooperative task and save a checkpoint.

Whole episodes are replayed through ExpoComm's local and message processors.
Per-agent chosen Q-values use masked double-Q QMIX targets. The algorithm file
owns both paper grounding objectives: global-state reconstruction and contrastive
same-timestep message learning.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from marl_envs import make_env
from modmarl.algorithms.expocomm import ExpoCommAgent, exponential_offsets
from modmarl.common.replay import EpisodeReplayBuffer

GRAD_CLIP = 10.0


class _RunningRewardStats:
    """Released parallel-variance update, restricted to valid replay steps."""

    def __init__(self, device: torch.device) -> None:
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = 1e-4

    def update(self, rewards: torch.Tensor, mask: torch.Tensor) -> None:
        values = rewards[mask.bool()]
        batch_count = values.numel()
        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        second = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / total
        )
        self.mean, self.var, self.count = new_mean, second / total, total


def _trainable(agent) -> list:
    """Online parameters: the recurrent network plus the mixer when one is configured."""
    parameters = list(agent.network.parameters())
    if agent.mixer is not None:
        parameters += list(agent.mixer.parameters())
    return parameters


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 1000,
    seed: int = 7,
    lr: float = 1e-4,
    hidden_dim: int = 64,
    mixer_hidden_dim: int = 64,
    use_mixer: bool = True,
    attention_dim: int = 16,
    buffer_episodes: int = 2000,
    batch_episodes: int = 64,
    warmup_episodes: int = 32,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_anneal_steps: int = 5_000,
    gamma: float = 0.95,
    aux_coef: float = 0.1,
    standardize_rewards: bool = True,
    updates_per_episode: int = 1,
    target_update_every: int = 200,
    evaluation_episodes: int = 20,
    topology: str = "one_peer",
    grounding: str = "contrastive",
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    n, obs_dim, num_actions = environment.n_agents, environment.obs_dim, environment.num_actions
    agent = ExpoCommAgent(
        n, obs_dim, num_actions, hidden_dim, mixer_hidden_dim,
        attention_dim=attention_dim,
        topology=topology, grounding=grounding, use_mixer=use_mixer,
    ).to(dev)
    optimizer = torch.optim.Adam(
        _trainable(agent), lr=lr,
    )
    replay = EpisodeReplayBuffer(buffer_episodes, environment.horizon, n, obs_dim)
    reward_stats = _RunningRewardStats(dev)

    evaluation_seed = seed + 100_000
    initial = _evaluate(
        agent, environment, num_actions, evaluation_seed, evaluation_episodes, dev,
    )
    random = _evaluate(
        agent, environment, num_actions, evaluation_seed, evaluation_episodes, dev,
        random_policy=True,
    )
    returns: list[float] = []
    total_steps = 0
    for episode in range(episodes):
        fraction = min(1.0, total_steps / epsilon_anneal_steps)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        episode_return = _collect_episode(
            agent, environment, replay, epsilon, num_actions, seed + episode, dev,
        )
        returns.append(episode_return)
        total_steps += environment.horizon
        if len(replay) >= max(batch_episodes, warmup_episodes):
            for _ in range(updates_per_episode):
                _update(
                    agent, optimizer, replay.sample(batch_episodes, dev), gamma, aux_coef,
                    standardize_rewards=standardize_rewards, reward_stats=reward_stats,
                )
        if episode > 0 and episode % target_update_every == 0:
            agent.update_targets()
        if episode % 20 == 0:
            print(f"episode {episode:4d}  eps {epsilon:4.2f}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(agent.state_dict(), checkpoint)
    final = _evaluate(
        agent, environment, num_actions, evaluation_seed, evaluation_episodes, dev,
    )
    evaluation_returns = final["returns"]
    return {
        "algorithm": "expocomm",
        "env": env,
        "episodes": episodes,
        "n_agents": n,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "evaluation_returns": evaluation_returns,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial,
        "random_evaluation": random,
        "final_evaluation": final,
        "communication_rate": _topology_rate(topology, n),
        "validation_criterion": {
            "minimum_improvement_fraction_of_absolute_random_return": 0.2,
            "scope": "every fixed seed",
        },
        "checkpoint": checkpoint,
    }


def _topology_rate(topology: str, n_agents: int) -> float:
    """Fraction of possible inter-agent links used by the selected topology."""
    if n_agents <= 1:
        return 0.0
    links_per_receiver = 1 if topology == "one_peer" else len(exponential_offsets(n_agents)) - 1
    return links_per_receiver / (n_agents - 1)


@torch.no_grad()
def _collect_episode(agent, environment, replay, epsilon, num_actions, episode_seed, device) -> float:
    obs, _ = environment.reset(seed=episode_seed)
    hidden, messages = agent.init_recurrent(1, device)
    previous_actions = torch.zeros(1, environment.n_agents, num_actions, device=device)
    observations, actions, rewards, dones = [obs], [], [], []
    episode_return = 0.0
    for timestep in range(environment.horizon):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values, hidden, messages = agent.step(
            obs_tensor, hidden, messages, timestep, previous_actions=previous_actions,
        )
        greedy = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()
        explore = np.random.random(environment.n_agents) < epsilon
        action = np.where(
            explore,
            np.random.randint(0, num_actions, size=environment.n_agents),
            greedy,
        ).astype(np.int64)
        previous_actions = F.one_hot(
            torch.as_tensor(action, device=device), num_classes=num_actions,
        ).to(dtype=obs_tensor.dtype).unsqueeze(0)
        next_obs, reward, terminated, truncated, _ = environment.step(action)
        observations.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(terminated))
        obs = next_obs
        episode_return += reward
        if terminated or truncated:
            break
    if replay is not None:
        replay.add_episode(
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
    return episode_return


@torch.no_grad()
def _evaluate(
    agent, environment, num_actions, seed, episodes, device, *, random_policy=False,
):
    rng_state = np.random.get_state()
    was_training = agent.training
    agent.eval()
    np.random.seed(seed)
    try:
        returns = [
            _collect_episode(
                agent, environment, None, 1.0 if random_policy else 0.0,
                num_actions, seed + episode, device,
            )
            for episode in range(episodes)
        ]
    finally:
        np.random.set_state(rng_state)
        agent.train(was_training)
    return {
        "returns": returns,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
    }


def _update(
    agent,
    optimizer,
    batch,
    gamma: float,
    aux_coef: float,
    *,
    standardize_rewards: bool = True,
    reward_stats: _RunningRewardStats | None = None,
) -> dict[str, float]:
    batch_size, horizon = batch.actions.shape[:2]
    n_agents = agent.n_agents
    hidden, messages = agent.init_recurrent(batch_size, batch.obs.device)
    target_hidden, target_messages = agent.init_recurrent(batch_size, batch.obs.device)
    online_q, target_q, online_messages = [], [], []
    for timestep in range(horizon + 1):
        if timestep == 0:
            previous_actions = batch.obs.new_zeros(batch_size, n_agents, agent.action_dim)
        else:
            previous_actions = F.one_hot(
                batch.actions[:, timestep - 1].long(), num_classes=agent.action_dim,
            ).to(dtype=batch.obs.dtype)
        q_values, hidden, messages = agent.step(
            batch.obs[:, timestep], hidden, messages, timestep,
            previous_actions=previous_actions,
        )
        online_q.append(q_values)
        online_messages.append(messages)
        with torch.no_grad():
            target_values, target_hidden, target_messages = agent.step(
                batch.obs[:, timestep], target_hidden, target_messages, timestep,
                previous_actions=previous_actions, target=True,
            )
            target_q.append(target_values)
    online_q = torch.stack(online_q, dim=1)
    target_q = torch.stack(target_q, dim=1)

    chosen = online_q[:, :-1].gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    states = batch.obs.reshape(batch_size, horizon + 1, -1)
    q_tot = agent.mix(chosen, states[:, :-1])
    with torch.no_grad():
        next_actions = online_q[:, 1:].argmax(dim=-1, keepdim=True)
        next_q = target_q[:, 1:].gather(-1, next_actions).squeeze(-1)
        next_tot = agent.mix(next_q, states[:, 1:], target=True)
        rewards = batch.rewards
        if q_tot.dim() == 3:                      # IDQN: one TD target per agent
            rewards = rewards.unsqueeze(-1)
        if standardize_rewards:
            if reward_stats is None:
                reward_stats = _RunningRewardStats(batch.obs.device)
            reward_stats.update(rewards, batch.mask)
            rewards = (rewards - reward_stats.mean) / torch.sqrt(reward_stats.var + 1e-8)
        targets = rewards + gamma * (1.0 - batch.dones) * next_tot
    mask = batch.mask.unsqueeze(-1) if q_tot.dim() == 3 else batch.mask
    td_error = (q_tot - targets) * mask
    td_loss = td_error.pow(2).sum() / mask.sum()

    message_sequence = torch.stack(online_messages[:-1], dim=1)
    aux_loss = agent.grounding_loss(
        message_sequence,
        states=states[:, :-1] if agent.grounding == "state" else None,
        mask=batch.mask,
    )

    if agent.grounding == "contrastive":
        auxiliary_scale = (td_loss / (aux_loss + 1e-10)).abs().detach()
        loss = td_loss + aux_coef * auxiliary_scale * aux_loss
    else:
        loss = td_loss + aux_coef * aux_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        _trainable(agent), GRAD_CLIP,
    )
    optimizer.step()
    return {"loss": float(loss.detach()), "td_loss": float(td_loss.detach()), "aux_loss": float(aux_loss.detach())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ExpoComm on a cooperative task.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--topology", choices=("one_peer", "static"), default="one_peer")
    parser.add_argument("--grounding", choices=("state", "contrastive"), default="contrastive")
    parser.add_argument("--checkpoint", default="expocomm.pt")
    args = parser.parse_args()
    train(
        env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed,
        topology=args.topology, grounding=args.grounding, checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
