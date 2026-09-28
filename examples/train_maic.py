"""Train MAIC on the released three-agent Hallway task.

The runner owns episode collection, replay unrolling, evaluation, and the released
RMSProp schedule. Equation 1 is supervised by the executed replay action. The pinned
release config uses QMIX and feeds the previous action to the controller; those are the
defaults here. (``join1.yaml`` disables ``obs_last_action`` only under ``env_args``,
where it is dead config -- the controller reads the top-level ``default.yaml`` value.)
The paper instead reports VDN for Hallway, so ``mixer="vdn"`` remains
an explicit paper configuration rather than being silently conflated with the release.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.maic import MAICAgent
from modmarl.common.replay import EpisodeReplayBuffer

GRAD_CLIP = 10.0
PRUNE_DELTA = 0.25


def train(*, env: str = "maic_hallway", n_agents: int = 3, horizon: int = 20,
          episodes: int = 100_000, seed: int = 7, lr: float = 5e-4,
          hidden_dim: int = 64, latent_dim: int = 8, attention_dim: int = 32,
          mixer_hidden_dim: int = 32, mixer: str = "qmix", include_previous_action: bool = True,
          buffer_episodes: int = 5000,
          batch_episodes: int = 32, warmup_episodes: int = 32,
          epsilon_start: float = 1.0, epsilon_end: float = 0.05,
          epsilon_anneal_steps: int = 50_000, gamma: float = 0.99,
          mi_loss_weight: float = 0.001, entropy_loss_weight: float = 0.01,
          updates_per_episode: int = 1, target_update_every: int = 200,
          evaluation_episodes: int = 300, device: str = "cpu",
          checkpoint: str | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    n, obs_dim, num_actions = environment.n_agents, environment.obs_dim, environment.num_actions
    agent = MAICAgent(n, obs_dim, num_actions, hidden_dim, latent_dim, attention_dim,
                      mixer_hidden_dim, mixer=mixer,
                      include_previous_action=include_previous_action).to(dev)
    parameters = list(agent.network.parameters())
    if agent.mixer is not None:
        parameters += list(agent.mixer.parameters())
    optimizer = torch.optim.RMSprop(parameters, lr=lr, alpha=0.99, eps=1e-5)
    replay = EpisodeReplayBuffer(buffer_episodes, environment.horizon, n, obs_dim)

    evaluation_seed = seed + 100_000
    initial = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    random = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes,
                       dev, random_policy=True)
    returns: list[float] = []
    training_metrics: list[dict[str, float | int]] = []
    total_steps = 0
    for episode in range(episodes):
        fraction = min(1.0, total_steps / max(1, epsilon_anneal_steps))
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        episode_return, steps = _collect_episode(
            agent, environment, replay, epsilon, num_actions, seed + episode, dev,
        )
        total_steps += steps
        returns.append(episode_return)
        if len(replay) >= max(batch_episodes, warmup_episodes):
            for _ in range(updates_per_episode):
                metrics = _update(agent, optimizer, replay.sample(batch_episodes, dev), gamma,
                                  mi_loss_weight, entropy_loss_weight)
                if (episode + 1) % 1000 == 0:
                    training_metrics.append({"episode": episode + 1, **metrics})
        if (episode + 1) % target_update_every == 0:
            agent.update_targets()
        if episode % 1000 == 0:
            print(f"episode {episode:6d}  eps {epsilon:4.2f}  return {episode_return:5.1f}", flush=True)

    if checkpoint is not None:
        torch.save(agent.state_dict(), checkpoint)
    final = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    ablated = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes,
                        dev, ablate_messages=True)
    return {
        "algorithm": "maic", "env": env, "episodes": episodes, "n_agents": n,
        "mixer": mixer, "include_previous_action": include_previous_action,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0, "returns": returns,
        "training_metrics": training_metrics,
        "checkpoint": checkpoint,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial, "random_evaluation": random,
        "final_evaluation": final, "message_ablated_evaluation": ablated,
        "communication_rate": final["communication_rate"],
        "validation_criterion": {"minimum_win_rate": 0.8,
                                 "scope": "every 300-episode confirmation seed"},
    }


@torch.no_grad()
def _collect_episode(agent, environment, replay, epsilon, num_actions, episode_seed, dev):
    obs, _ = environment.reset(seed=episode_seed)
    hidden = agent.init_hidden(1, dev)
    previous = agent.initial_previous_actions(1, dev)
    obs_list, actions, rewards, dones = [obs], [], [], []
    episode_return = 0.0
    for step_index in range(environment.horizon):  # noqa: B007  (read after the loop)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        result = agent.step(obs_tensor, previous, hidden)
        hidden = result.hidden
        greedy = result.q.argmax(-1).squeeze(0).cpu().numpy()
        explore = np.random.random(environment.n_agents) < epsilon
        action = np.where(explore, np.random.randint(num_actions, size=environment.n_agents), greedy)
        action = action.astype(np.int64)
        previous = torch.nn.functional.one_hot(torch.as_tensor(action, device=dev), num_actions).float().unsqueeze(0)
        next_obs, reward, terminated, truncated, _ = environment.step(action)
        obs_list.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(terminated or truncated))
        obs = next_obs
        episode_return += reward
        if terminated or truncated:
            break
    replay.add_episode(obs=np.asarray(obs_list, dtype=np.float32),
                       actions=np.asarray(actions, dtype=np.int64),
                       rewards=np.asarray(rewards, dtype=np.float32),
                       dones=np.asarray(dones, dtype=np.float32))
    return episode_return, step_index + 1


def _update(agent, optimizer, batch, gamma, mi_loss_weight, entropy_loss_weight) -> dict[str, float]:
    batch_size, horizon = batch.actions.shape[:2]
    dev = batch.obs.device
    hidden = agent.init_hidden(batch_size, dev)
    target_hidden = agent.init_hidden(batch_size, dev)
    previous = agent.initial_previous_actions(batch_size, dev)
    online_q, target_q, mi_losses, sparsity_losses = [], [], [], []
    for time in range(horizon + 1):
        step = agent.step(batch.obs[:, time], previous, hidden)
        hidden = step.hidden
        online_q.append(step.q)
        with torch.no_grad():
            target_step = agent.step(batch.obs[:, time], previous, target_hidden,
                                     target=True)
            target_hidden = target_step.hidden
            target_q.append(target_step.q)
        if time < horizon:
            mi_losses.append(agent.teammate_model_loss(step, batch.actions[:, time]))
            sparsity_losses.append(agent.sparsity_loss(step))
            previous = torch.nn.functional.one_hot(batch.actions[:, time], agent.action_dim).float()
    online_q = torch.stack(online_q, dim=1)
    target_q = torch.stack(target_q, dim=1)
    chosen = online_q[:, :-1].gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    state = batch.obs.reshape(batch_size, horizon + 1, -1)
    q_total = agent.mix(chosen, state[:, :-1])
    with torch.no_grad():
        next_actions = online_q[:, 1:].argmax(-1, keepdim=True)
        next_values = target_q[:, 1:].gather(-1, next_actions).squeeze(-1)
        target_total = agent.mix(next_values, state[:, 1:], target=True)
        targets = batch.rewards + gamma * (1.0 - batch.dones) * target_total
    td_loss = (((q_total - targets) * batch.mask) ** 2).sum() / batch.mask.sum()
    mi_loss = (torch.stack(mi_losses, 1) * batch.mask).sum() / batch.mask.sum()
    sparsity_loss = (torch.stack(sparsity_losses, 1) * batch.mask).sum() / batch.mask.sum()
    loss = td_loss + mi_loss_weight * mi_loss + entropy_loss_weight * sparsity_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = list(agent.network.parameters())
    if agent.mixer is not None:
        parameters += list(agent.mixer.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
    optimizer.step()
    return {
        "td_loss": float(td_loss.detach()),
        "teammate_model_loss": float(mi_loss.detach()),
        "sparsity_loss": float(sparsity_loss.detach()),
        "q_mean": float(q_total.detach().mean()),
        "target_mean": float(targets.detach().mean()),
        "gradient_norm": float(grad_norm.detach()),
    }


@torch.no_grad()
def _evaluate(agent, env_name, n_agents, horizon, seed, episodes, device, *,
              random_policy=False, ablate_messages=False):
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes, communication_rates = [], [], []
    was_training = agent.training
    agent.eval()
    try:
        for episode in range(episodes):
            obs, _ = environment.reset(seed=seed + episode)
            hidden = agent.init_hidden(1, device)
            previous = agent.initial_previous_actions(1, device)
            episode_return, info, rates = 0.0, {}, []
            for _ in range(environment.horizon):
                if random_policy:
                    action = generator.integers(environment.num_actions, size=environment.n_agents)
                    rates.append(0.0)
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    threshold = float("inf") if ablate_messages else PRUNE_DELTA
                    result = agent.step(obs_tensor, previous, hidden, deterministic=True,
                                        prune_threshold=threshold)
                    hidden = result.hidden
                    action = result.q.argmax(-1).squeeze(0).cpu().numpy()
                    rates.append(float((result.alpha > 0).float().sum() /
                                       (environment.n_agents * (environment.n_agents - 1))))
                    previous = torch.nn.functional.one_hot(
                        torch.as_tensor(action, device=device), environment.num_actions,
                    ).float().unsqueeze(0)
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            successes.append(float(bool(info.get("success", False))))
            communication_rates.append(float(np.mean(rates)))
    finally:
        environment.close()
        agent.train(was_training)
    return {"returns": returns, "successes": successes,
            "win_rate": float(np.mean(successes)) if successes else 0.0,
            "communication_rates": communication_rates,
            "communication_rate": float(np.mean(communication_rates)) if communication_rates else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAIC on the paper Hallway task.")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="maic.pt")
    args = parser.parse_args()
    train(episodes=args.episodes, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
