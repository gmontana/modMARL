"""Train paper-faithful CACOM with QMIX on a cooperative discrete task.

The ordinary update unrolls CACOM's two communication stages and adds the
helper-value prediction loss. Gate parameters are excluded and updated separately
from released-code counterfactual labels before their delayed deployment. The
released auxiliary weight is 0.1. Environment-step schedules count actual steps,
including episodes that terminate before their configured horizon.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from marl_envs import make_env
from modmarl.algorithms.cacom import CACOMAgent
from modmarl.common.replay import EpisodeReplayBuffer

GRAD_CLIP = 10.0


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 1000,
    seed: int = 7,
    lr: float = 5e-4,
    gate_lr: float = 1e-4,
    hidden_dim: int = 64,
    encode_dim: int = 8,
    request_dim: int = 4,
    response_dim: int = 8,
    bits: int = 2,
    mixer_hidden_dim: int = 32,
    buffer_episodes: int = 5000,
    batch_episodes: int = 32,
    warmup_episodes: int = 32,
    gate_start_steps: int = 200_000,
    gate_update_every_steps: int = 10_000,
    gate_label_mode: str = "release",
    target_update_every: int = 200,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_anneal_steps: int = 50_000,
    gamma: float = 0.99,
    auxiliary_weight: float = 0.1,
    evaluation_episodes: int = 20,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    input_dim = environment.obs_dim + environment.num_actions + environment.n_agents
    entity_schema = _entity_schema(env, environment.n_agents, environment.num_actions)
    agent = CACOMAgent(
        environment.n_agents, environment.obs_dim, environment.num_actions,
        input_dim=input_dim, entity_schema=entity_schema,
        hidden_dim=hidden_dim, encode_dim=encode_dim, request_dim=request_dim,
        response_dim=response_dim, bits=bits, mixer_hidden_dim=mixer_hidden_dim,
    ).to(dev)
    policy_optimizer = torch.optim.RMSprop(
        agent.policy_parameters(), lr=lr, alpha=0.99, eps=1e-5,
    )
    gate_optimizer = _gate_optimizer(agent.gate_parameters(), gate_lr)
    replay = EpisodeReplayBuffer(
        buffer_episodes, environment.horizon, environment.n_agents, environment.obs_dim,
    )
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agent, environment, evaluation_episodes, evaluation_seed, dev,
        epsilon=0.0, force_all_links=True,
    )
    random_evaluation = _evaluate(
        agent, environment, evaluation_episodes, evaluation_seed, dev,
        epsilon=1.0, force_all_links=True,
    )

    returns: list[float] = []
    total_steps = 0
    gate_rng = np.random.default_rng(seed)
    last_gate_update_step = -gate_update_every_steps
    gate_updates = 0
    for episode in range(episodes):
        fraction = min(1.0, total_steps / max(1, epsilon_anneal_steps))
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        force_all_links = total_steps < gate_start_steps
        episode_return, episode_steps, _, _ = _collect_episode(
            agent, environment, replay, epsilon, environment.num_actions,
            seed + episode, dev, force_all_links,
        )
        total_steps += episode_steps
        returns.append(episode_return)
        if len(replay) >= max(batch_episodes, warmup_episodes):
            batch = replay.sample(batch_episodes, dev)
            _update_policy(
                agent, policy_optimizer, batch, gamma, auxiliary_weight,
                force_all_links=force_all_links,
            )
            if total_steps - last_gate_update_step >= gate_update_every_steps:
                _update_gate(
                    agent, gate_optimizer, batch,
                    helper=int(gate_rng.integers(agent.n_agents)),
                    force_keep=force_all_links,
                    mode=gate_label_mode,
                )
                last_gate_update_step = total_steps
                gate_updates += 1
        if episode > 0 and episode % target_update_every == 0:
            agent.update_targets()
        if episode % 20 == 0:
            print(f"episode {episode:4d}  eps {epsilon:4.2f}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(agent.state_dict(), checkpoint)
    final_evaluation = _evaluate(
        agent, environment, evaluation_episodes, evaluation_seed, dev,
        epsilon=0.0, force_all_links=total_steps < gate_start_steps,
    )
    return {
        "algorithm": "cacom", "env": env, "episodes": episodes,
        "n_agents": environment.n_agents,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "returns": returns,
        "evaluation_returns": final_evaluation["returns"],
        "communication_rate": final_evaluation["communication_rate"],
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "validation_criterion": {
            "return_improvement_fraction_of_abs_random": 0.2,
            "scope": "every confirmation seed",
        },
        "total_steps": total_steps,
        "gate_updates": gate_updates, "checkpoint": checkpoint,
        "gate_label_mode": gate_label_mode,
    }


def _gate_optimizer(parameters, learning_rate: float) -> torch.optim.Adam:
    """Paper Appendix B optimizer for the separately supervised local gate."""
    return torch.optim.Adam(parameters, lr=learning_rate)


@torch.no_grad()
def _evaluate(agent, environment, episodes, seed, device, *, epsilon, force_all_links):
    rng_state = np.random.get_state()
    was_training = agent.training
    agent.eval()
    np.random.seed(seed)
    try:
        evaluations = [
            _collect_episode(
                agent, environment, None, epsilon, environment.num_actions,
                seed + index, device, force_all_links,
            )
            for index in range(episodes)
        ]
    finally:
        np.random.set_state(rng_state)
        agent.train(was_training)
    returns = [evaluation[0] for evaluation in evaluations]
    communication_rates = [evaluation[2] for evaluation in evaluations]
    return {
        "returns": returns,
        "successes": [float(bool(evaluation[3].get("success", False))) for evaluation in evaluations],
        "mean_distances": [float(evaluation[3].get("mean_distance", float("nan")))
                           for evaluation in evaluations],
        "mean_return": float(np.mean(returns)),
        "communication_rates": communication_rates,
        "communication_rate": float(np.mean(communication_rates)),
    }


@torch.no_grad()
def _collect_episode(agent, environment, replay, epsilon, num_actions, episode_seed, device, force_all_links):
    obs, _ = environment.reset(seed=episode_seed)
    hidden = agent.init_hidden(1, device)
    previous_actions = torch.zeros(1, environment.n_agents, num_actions, device=device)
    agent_ids = torch.eye(environment.n_agents, device=device).unsqueeze(0)
    observations, actions, rewards, dones = [obs], [], [], []
    episode_return = 0.0
    steps = 0
    active_links = 0.0
    possible_links = 0
    info = {}
    for _ in range(environment.horizon):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        inputs = torch.cat([obs_tensor, previous_actions, agent_ids], dim=-1)
        q_values, hidden, products = agent.network.step(
            inputs, hidden, force_all_links=force_all_links,
        )
        active_links += float(products["gate_mask"].sum())
        possible_links += environment.n_agents * (environment.n_agents - 1)
        greedy = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()
        explore = np.random.random(environment.n_agents) < epsilon
        action = np.where(explore, np.random.randint(0, num_actions, environment.n_agents), greedy)
        previous_actions = F.one_hot(
            torch.as_tensor(action, device=device), num_classes=num_actions,
        ).to(dtype=obs_tensor.dtype).unsqueeze(0)
        next_obs, reward, terminated, truncated, info = environment.step(action)
        observations.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(terminated))
        obs = next_obs
        episode_return += reward
        steps += 1
        if terminated or truncated:
            break
    if replay is not None:
        replay.add_episode(
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
    return episode_return, steps, active_links / max(1, possible_links), info


def _unroll(network, obs, actions, action_dim, *, force_all_links):
    hidden = network.init_hidden(obs.shape[0], obs.device)
    agent_ids = torch.eye(network.n_agents, device=obs.device).unsqueeze(0).expand(
        obs.shape[0], -1, -1,
    )
    q_values, auxiliary = [], []
    for timestep in range(obs.shape[1]):
        previous_actions = obs.new_zeros(obs.shape[0], network.n_agents, action_dim)
        if timestep > 0:
            previous_actions = F.one_hot(
                actions[:, timestep - 1].long(), num_classes=action_dim,
            ).to(dtype=obs.dtype)
        inputs = obs[:, timestep]
        if network.obs_dim != inputs.shape[-1]:
            inputs = torch.cat([inputs, previous_actions, agent_ids], dim=-1)
        q, hidden, products = network.step(
            inputs, hidden, force_all_links=force_all_links,
        )
        q_values.append(q)
        auxiliary.append(network.helper_value_loss(products, q))
    return torch.stack(q_values, dim=1), torch.stack(auxiliary)


def _update_policy(agent, optimizer, batch, gamma, auxiliary_weight, *, force_all_links):
    batch_size, horizon = batch.actions.shape[:2]
    online_q, auxiliary = _unroll(
        agent.network, batch.obs, batch.actions, agent.action_dim,
        force_all_links=force_all_links,
    )
    with torch.no_grad():
        target_q, _ = _unroll(
            agent.target_network, batch.obs, batch.actions, agent.action_dim,
            force_all_links=force_all_links,
        )
    chosen = online_q[:, :-1].gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    states = batch.obs.reshape(batch_size, horizon + 1, -1)
    q_total = agent.mixer(
        chosen.reshape(-1, agent.n_agents), states[:, :-1].reshape(-1, states.shape[-1]),
    ).view(batch_size, horizon)
    with torch.no_grad():
        next_actions = online_q[:, 1:].argmax(dim=-1, keepdim=True)
        next_q = target_q[:, 1:].gather(-1, next_actions).squeeze(-1)
        next_total = agent.target_mixer(
            next_q.reshape(-1, agent.n_agents), states[:, 1:].reshape(-1, states.shape[-1]),
        ).view(batch_size, horizon)
        targets = batch.rewards + gamma * (1.0 - batch.dones) * next_total
    td_error = (q_total - targets) * batch.mask
    td_loss = td_error.pow(2).sum() / batch.mask.sum().clamp_min(1.0)
    time_weights = torch.cat([batch.mask, batch.mask[:, -1:]], dim=1)
    aux_loss = (auxiliary.transpose(0, 1) * time_weights).sum() / time_weights.sum().clamp_min(1.0)
    loss = td_loss + auxiliary_weight * aux_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy_parameters(), GRAD_CLIP)
    optimizer.step()
    return {"loss": float(loss.detach()), "td_loss": float(td_loss.detach()), "aux_loss": float(aux_loss.detach())}


def _update_gate(
    agent, optimizer, batch, helper: int, threshold: float = 0.0, *, force_keep: bool = False,
    mode: str = "release",
):
    hidden = agent.init_hidden(batch.obs.shape[0], batch.obs.device)
    agent_ids = torch.eye(agent.n_agents, device=batch.obs.device).unsqueeze(0).expand(
        batch.obs.shape[0], -1, -1,
    )
    losses = []
    for timestep in range(batch.obs.shape[1] - 1):
        previous_actions = batch.obs.new_zeros(
            batch.obs.shape[0], agent.n_agents, agent.action_dim,
        )
        if timestep > 0:
            previous_actions = F.one_hot(
                batch.actions[:, timestep - 1].long(), num_classes=agent.action_dim,
            ).to(dtype=batch.obs.dtype)
        inputs = batch.obs[:, timestep]
        if agent.network.obs_dim != inputs.shape[-1]:
            inputs = torch.cat([inputs, previous_actions, agent_ids], dim=-1)
        if mode == "release":
            logits, labels = agent.network.gate_labels(inputs, hidden, helper, threshold)
        else:
            logits, labels = agent.network.gate_labels(
                inputs, hidden, helper, threshold, mode=mode,
                mixer=agent.mixer, state=batch.obs[:, timestep].flatten(1),
            )
        if force_keep:
            labels = torch.zeros_like(labels)
        losses.append(F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1)))
        with torch.no_grad():
            _, hidden, _ = agent.network.step(inputs, hidden)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.gate_parameters(), GRAD_CLIP)
    optimizer.step()
    return float(loss.detach())


def _entity_schema(
    env: str, n_agents: int, action_dim: int,
) -> tuple[tuple[int, int], ...] | None:
    """Entity types for the built-in navigation observation plus controller inputs.

    ``(count, length)`` per type, as the release's ``obs_segs``: the landmarks share one
    encoder and the other agents share another, so the encoder is permutation-equivariant
    over each interchangeable group.
    """
    if env != "navigation":
        return None
    schema = (
        (1, 2),                 # own position
        (1, 2),                 # own velocity
        (n_agents, 2),          # landmarks
        (n_agents - 1, 2),      # other agents
        (1, action_dim),        # previous action
        (1, n_agents),          # agent identity
    )
    return tuple((count, length) for count, length in schema if count > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CACOM on a cooperative task.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="cacom.pt")
    args = parser.parse_args()
    train(
        env=args.env, episodes=args.episodes, n_agents=args.n_agents,
        seed=args.seed, checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
