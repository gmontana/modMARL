"""Train SchedNet on a cooperative task and save a checkpoint.

The released selector is a stochastic policy trained from the TD error of ``V(s)``;
the deterministic weight generator ascends ``Q(s,w)``. Both values share the critic's
first two state layers. Replay records the executed schedule priorities, and deterministic
evaluation removes both epsilon exploration and policy sampling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.schednet import SchedNetAgent, top_k_schedule
from modmarl.common.replay import ReplayBuffer

GAMMA = 0.9
TAU = 0.05
ENTROPY_COEF = 0.01        # released `ac_network.py`: 0.01 * entropy
EPSILON_START = 0.5
EPSILON_FINAL = 0.1
EPSILON_DECAY_STEPS = 750_000


@dataclass
class SchedNetBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor   # (batch, n_agents): the scheduling weights emitted that step


class SchedNetReplayBuffer(ReplayBuffer):
    """Standard MADDPG replay extended with the per-step scheduling weights — the weight
    generators' continuous 'actions', stored so the weight critic can be trained off-policy."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int) -> None:
        super().__init__(capacity, n_agents, obs_dim)
        self.weights = np.zeros((capacity, n_agents), dtype=np.float32)

    def add(self, *, obs, actions, reward, next_obs, done, weights) -> None:
        self.weights[self.ptr] = weights            # written at ptr before super() advances it
        super().add(obs, actions, reward, next_obs, done)

    def sample(self, batch_size: int, device: torch.device) -> SchedNetBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        return SchedNetBatch(
            obs=torch.as_tensor(self.obs[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
            weights=torch.as_tensor(self.weights[indices], device=device),
        )


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    bandwidth: int | None = None,
    actor_learning_rate: float = 1e-5,
    weight_learning_rate: float = 1e-5,
    critic_learning_rate: float = 1e-4,
    actor_hidden_dim: int = 32,
    critic_hidden_dim: int = 64,
    scheduler_hidden_dim: int = 32,
    message_dim: int = 2,
    buffer_size: int = 10_000,
    batch_size: int = 64,
    warmup_steps: int = 640,
    epsilon_decay_steps: int = EPSILON_DECAY_STEPS,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    n = environment.n_agents
    k = bandwidth if bandwidth is not None else max(1, n // 2)   # scheduling bandwidth
    agent = SchedNetAgent(
        n_agents=n, obs_dim=environment.obs_dim, action_dim=environment.num_actions,
        message_dim=message_dim,
        actor_hidden_dim=actor_hidden_dim,
        critic_hidden_dim=critic_hidden_dim,
        scheduler_hidden_dim=scheduler_hidden_dim,
        bandwidth=k,
    ).to(dev)

    action_opt = torch.optim.Adam(
        chain(agent.message_encoder.parameters(), agent.action_selector.parameters()),
        lr=actor_learning_rate,
    )
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=critic_learning_rate)
    weight_opt = torch.optim.Adam(
        agent.weight_generator.parameters(), lr=weight_learning_rate,
    )
    replay = SchedNetReplayBuffer(buffer_size, n, environment.obs_dim)

    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev, k,
    )
    random_evaluation = _evaluate(
        agent,
        env,
        n_agents,
        horizon,
        evaluation_seed,
        evaluation_episodes,
        dev,
        k,
        random_policy=True,
    )

    total_steps = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        for _ in range(environment.horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)   # (1, n, obs_dim)
            epsilon = max(
                EPSILON_FINAL,
                EPSILON_START - total_steps / max(1, epsilon_decay_steps),
            )
            random_schedule = total_steps < warmup_steps or np.random.random() < epsilon
            priorities = (
                torch.rand((1, n), dtype=torch.float32, device=dev)
                if random_schedule
                else None
            )
            with torch.no_grad():
                action_idx, _, weights, _ = agent.act(obs_t, k, priorities=priorities)
            if total_steps < warmup_steps or np.random.random() < epsilon:
                action = np.random.randint(0, environment.num_actions, size=n)
            else:
                action = action_idx.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = environment.step(action)
            replay.add(
                obs=obs, actions=action, reward=reward, next_obs=next_obs, done=terminated,
                weights=weights.squeeze(0).cpu().numpy(),
            )
            obs = next_obs
            episode_return += reward
            total_steps += 1

            if len(replay) >= batch_size and total_steps >= warmup_steps:
                _update(
                    agent,
                    action_opt,
                    critic_opt,
                    weight_opt,
                    replay.sample(batch_size, dev),
                    k,
                )

            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(agent.state_dict(), checkpoint)

    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev, k,
    )
    return {
        "algorithm": "schednet",
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
        "validation_criterion": {
            "return_improvement_fraction_of_abs_random": 0.2,
            "scope": "every confirmation seed",
        },
        "communication_rate": k / n,
        "config": {
            "env": env,
            "episodes": episodes,
            "seed": seed,
            "n_agents": n,
            "horizon": environment.horizon,
            "gamma": GAMMA,
            "tau": TAU,
            "entropy_coefficient": ENTROPY_COEF,
            "actor_learning_rate": actor_learning_rate,
            "weight_learning_rate": weight_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "actor_hidden_dim": actor_hidden_dim,
            "critic_hidden_dim": critic_hidden_dim,
            "scheduler_hidden_dim": scheduler_hidden_dim,
            "message_dim": message_dim,
            "bandwidth": k,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "epsilon_start": EPSILON_START,
            "epsilon_final": EPSILON_FINAL,
            "epsilon_decay_steps": epsilon_decay_steps,
        },
    }


def _update(agent, actor_opt, critic_opt, weight_opt, batch, k):
    """One released training step (`ac_network.py`, Algorithm 1 lines 12-17).

    The critic is a single network with a value head and a schedule head; both are fit by
    one joint loss. The action selector follows paper Equation (4) -- the stochastic policy
    gradient weighted by the value head's TD error, plus the release's 0.01 entropy bonus.
    The weight generator is deterministic and ascends the schedule head.
    """
    # --- Critic: V(s) and Q_sched(s, w), one joint loss over both TD errors. ---
    with torch.no_grad():
        next_weights = agent.target_weight_generator(batch.next_obs)
        next_value, next_schedule_value = agent.target_critic(batch.next_obs, next_weights)
        value_target = batch.rewards + GAMMA * (1.0 - batch.dones) * next_value
        continuation = 1.0 - batch.dones
        schedule_target = batch.rewards + GAMMA * continuation * next_schedule_value

    value, schedule_value = agent.critic(batch.obs, batch.weights)
    td_error = value_target - value
    critic_loss = (td_error.pow(2) + (schedule_target - schedule_value).pow(2)).mean()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()

    # --- Action selector + message encoder: paper Equation (4). The schedule is the stored
    # behaviour schedule; it belongs to the weight-generator level. ---
    schedule = top_k_schedule(batch.weights, k)
    broadcast = agent.broadcast_for(batch.obs, schedule)
    distribution = agent.action_selector.distribution(batch.obs, broadcast)
    log_prob = distribution.log_prob(batch.actions.long())
    joint_log_prob = log_prob.sum(dim=-1)
    joint_entropy = distribution.entropy().sum(dim=-1)
    actor_loss = -(joint_log_prob * td_error.detach() + ENTROPY_COEF * joint_entropy).mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()

    # --- Released WG chain rule: critic derivative at replayed priorities. ---
    # agent.py calls grads_for_scheduler(s, p), where p comes from replay,
    # then feeds that derivative to the current weight generator. Evaluating
    # the critic at current mu(o) instead is the paper's DDPG interpretation.
    agent.critic.requires_grad_(False)
    replay_priorities = batch.weights.detach().requires_grad_(True)
    _, schedule_value_for_weights = agent.critic(batch.obs, replay_priorities)
    priority_gradient = torch.autograd.grad(schedule_value_for_weights.sum(), replay_priorities)[0]
    weight_loss = -(agent.weight_generator(batch.obs) * priority_gradient.detach()).sum(-1).mean()
    weight_opt.zero_grad(set_to_none=True)
    weight_loss.backward()
    weight_opt.step()
    agent.critic.requires_grad_(True)

    agent.soft_update(TAU)

@torch.no_grad()
def _evaluate(
    agent: SchedNetAgent,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    bandwidth: int,
    *,
    random_policy: bool = False,
) -> dict:
    """Evaluate deterministic top-k routing and argmax actions without exploration."""
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes = [], []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            action = (
                generator.integers(
                    0, environment.num_actions, size=environment.n_agents,
                )
                if random_policy
                else agent.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    bandwidth,
                    deterministic=True,
                )[0].squeeze(0).cpu().numpy()
            )
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
    return {
        "returns": returns,
        "successes": successes,
        "communication_rate": bandwidth / environment.n_agents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SchedNet on a cooperative MPE task.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--bandwidth", type=int, default=None, help="number of agents scheduled per step (default n//2)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="schednet.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, bandwidth=args.bandwidth,
          seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
