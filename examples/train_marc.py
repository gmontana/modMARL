"""Train MARC (Multi-Agent Relational Critic) on the collaborative pick-and-place task.

Run:
    python examples/train_marc.py --episodes 300

Soft actor-critic where the critic reasons over a relational graph of the state
(node features + typed relations) through a relational GNN, with per-agent counterfactual
baselines. Requires the optional `macpp` environment (`pip install -e ".[macpp]"`).
`train()` is importable; built from the public modMARL API.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from marl_envs import MACPPEnv, macpp_available
from modmarl import MARCAgent, MARCRelationalCritic
from modmarl.common.replay import MARCReplayBuffer
from modmarl.components import soft_update_module

GAMMA = 0.99
TAU = 0.001
ALPHA = 0.01          # entropy temperature
POLICY_REG = 1e-3
GRAD_CLIP = 10.0          # critic; released `attention_sac.py` uses 10 * n_agents
ACTOR_GRAD_CLIP = 0.5     # released `attention_sac.py:232`
NORM_REWS = True          # released configs all set `norm_rews: true`


def train(
    *,
    n_agents: int = 2,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    grid_size: int = 5,
    n_pickers: int = 1,
    n_objects: int = 1,
    env_version: str = "v0",
    pi_lr: float = 1e-3,
    q_lr: float = 1e-3,
    actor_hidden_dim: int = 128,
    critic_hidden_dim: int = 128,
    embed_dim: int = 128,
    relational_layers: int = 1,
    buffer_size: int = 1_000_000,
    batch_size: int = 1024,
    warmup_steps: int = 1000,
    critic_weight_decay: float = 1e-3,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    if not macpp_available():
        raise ImportError('MARC requires the optional macpp environment: pip install -e ".[macpp]"')
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    env = MACPPEnv(
        grid_size=grid_size, n_agents=n_agents, n_pickers=n_pickers, n_objects=n_objects,
        horizon=horizon, version=env_version, seed=seed,
    )
    n = env.n_agents
    agents = [MARCAgent(obs_dim=env.obs_dim, action_dim=env.num_actions, hidden_dim=actor_hidden_dim).to(dev) for _ in range(n)]
    critic = MARCRelationalCritic(
        n_agents=n, node_feature_dim=env.node_feature_dim, action_dim=env.num_actions,
        hidden_dim=critic_hidden_dim, embed_dim=embed_dim, num_relations=env.num_relations,
        num_relational_layers=relational_layers,
    ).to(dev)
    target_critic = copy.deepcopy(critic)

    actor_opts = [torch.optim.Adam(a.actor.parameters(), lr=pi_lr) for a in agents]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=q_lr, weight_decay=critic_weight_decay)
    replay = MARCReplayBuffer(
        capacity=buffer_size, n_agents=n, obs_dim=env.obs_dim,
        n_entities=env.n_entities, node_feature_dim=env.node_feature_dim, num_relations=env.num_relations,
    )

    evaluation_seed = seed + 100_000
    initial = _evaluate(
        agents, evaluation_seed, evaluation_episodes, dev, grid_size, n, n_pickers,
        n_objects, horizon, env_version,
    )
    random = _evaluate(
        agents, evaluation_seed, evaluation_episodes, dev, grid_size, n, n_pickers,
        n_objects, horizon, env_version, random_policy=True,
    )
    total_steps = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        graph = env.graph_observation()
        episode_return = 0.0
        for _ in range(env.horizon):
            if total_steps < warmup_steps:
                action = np.random.randint(0, env.num_actions, size=n)
            else:
                action = _select_actions(agents, obs, dev)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_graph = env.graph_observation()
            replay.add(
                obs=obs, node_features=graph.node_features, relations=graph.relations,
                actions=action, reward=reward, next_obs=next_obs,
                next_node_features=next_graph.node_features, next_relations=next_graph.relations,
                done=terminated,
            )
            obs, graph = next_obs, next_graph
            episode_return += reward
            total_steps += 1

            if (
                len(replay) >= batch_size
                and total_steps >= warmup_steps
                and total_steps % 100 == 0
            ):
                for _ in range(4):
                    _update(
                        agents, critic, target_critic, actor_opts, critic_opt,
                        replay.sample(batch_size, dev, normalize_rewards=NORM_REWS), env.num_actions,
                    )

            if terminated or truncated:
                break
        returns.append(episode_return)
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    env.close()
    if checkpoint is not None:
        torch.save({"actors": [a.state_dict() for a in agents], "critic": critic.state_dict()}, checkpoint)
    final = _evaluate(
        agents, evaluation_seed, evaluation_episodes, dev, grid_size, n, n_pickers,
        n_objects, horizon, env_version,
    )

    return {
        "algorithm": "marc",
        "env": "macpp",
        "episodes": episodes,
        "n_agents": n,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial,
        "random_evaluation": random,
        "final_evaluation": final,
        "validation_criterion": {
            "minimum_improvement_fraction_of_absolute_random_return": 0.2,
            "scope": "every fixed seed",
        },
    }


@torch.no_grad()
def _select_actions(agents, obs, dev) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    actions = []
    for i, agent in enumerate(agents):
        training = agent.actor.training
        agent.actor.eval()
        _, action_idx, _, _, _, _, _ = agent.actor.sample(obs_t[i].unsqueeze(0))
        agent.actor.train(training)
        actions.append(int(action_idx.item()))
    return np.asarray(actions, dtype=np.int64)


@torch.no_grad()
def _evaluate(
    agents, seed, episodes, device, grid_size, n_agents, n_pickers, n_objects,
    horizon, env_version, *, random_policy=False,
):
    environment = MACPPEnv(
        grid_size=grid_size, n_agents=n_agents, n_pickers=n_pickers,
        n_objects=n_objects, horizon=horizon, version=env_version, seed=seed,
    )
    generator = np.random.default_rng(seed)
    returns, successes = [], []
    try:
        for episode in range(episodes):
            obs, _ = environment.reset(seed=seed + episode)
            episode_return, info = 0.0, {}
            for _ in range(environment.horizon):
                action = (
                    generator.integers(environment.num_actions, size=environment.n_agents)
                    if random_policy else _select_actions(agents, obs, device)
                )
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            successes.append(float(bool(info.get("success", False))))
    finally:
        environment.close()
    return {
        "returns": returns,
        "successes": successes,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
    }


def _update(agents, critic, target_critic, actor_opts, critic_opt, batch, num_actions):
    n = len(agents)
    onehot = F.one_hot(batch.actions.long(), num_actions).float()
    rewards = batch.rewards.unsqueeze(-1).expand(-1, n)
    dones = batch.dones.unsqueeze(-1).expand(-1, n)

    # Critic: soft TD target; Q is read off the relational graph (node features + typed relations).
    with torch.no_grad():
        next_actions, next_log_pis = [], []
        for i, agent in enumerate(agents):
            action_oh, _, _, _, _, chosen_log_prob, _ = agent.target_actor.sample(batch.next_obs[:, i])
            next_actions.append(action_oh)
            next_log_pis.append(chosen_log_prob)
        next_actions = torch.stack(next_actions, dim=1)
        next_log_pis = torch.stack(next_log_pis, dim=1)
        target_q, _ = target_critic(batch.next_node_features, batch.next_relations, next_actions, return_all_q=False)
        target_values = target_q - ALPHA * next_log_pis
        y = rewards + GAMMA * (1.0 - dones) * target_values

    q, _ = critic(batch.node_features, batch.relations, onehot, return_all_q=False)
    critic_loss = F.mse_loss(q, y)
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic.scale_shared_grads()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    critic_opt.step()

    # Actors: counterfactual advantage from the relational critic, per agent. The
    # gradient is score-function only — the advantage is a detached weight on
    # log-pi — so the critic is evaluated once, under no_grad, on the sampled joint.
    samples = [agent.actor.sample(batch.obs[:, i]) for i, agent in enumerate(agents)]
    with torch.no_grad():
        joint = torch.stack([s[0] for s in samples], dim=1)
        q_taken, all_q = critic(batch.node_features, batch.relations, joint, return_all_q=True)
    for i, agent in enumerate(agents):
        _, _, logits, probs, _, chosen_log_prob, _ = samples[i]
        baseline = (all_q[:, i] * probs).sum(dim=-1)
        advantage = q_taken[:, i] - baseline
        actor_loss = (chosen_log_prob * (ALPHA * chosen_log_prob - advantage).detach()).mean()
        actor_loss = actor_loss + POLICY_REG * logits.pow(2).mean()
        actor_opts[i].zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), ACTOR_GRAD_CLIP)
        actor_opts[i].step()

    for agent in agents:
        agent.soft_update(TAU)
    soft_update_module(target_critic, critic, TAU)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MARC on the collaborative pick-and-place task.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--n-pickers", type=int, default=1)
    parser.add_argument("--n-objects", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="marc.pt")
    args = parser.parse_args()
    train(
        episodes=args.episodes, n_agents=args.n_agents, grid_size=args.grid_size,
        n_pickers=args.n_pickers, n_objects=args.n_objects, seed=args.seed, checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
