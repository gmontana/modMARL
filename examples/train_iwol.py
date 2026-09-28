"""Train IWoL on bounded cooperative navigation.

The runner supplies joint recurrent rollouts, privileged reconstruction targets,
and a fully connected physical communication graph.  It validates Im-IWoL with
message-free actor-only execution; the critic communication protocol is invoked
only while collecting training values.  This is learning evidence for the
dependency-free library implementation, not a paper-scale robotics experiment.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marl_envs import make_env
from modmarl.algorithms.iwol import IWoLAgent, IWoLMode, IWoLRollout

SOURCE_REVISION = "de3bc5b1e50bd9c4d90672a6355269ea2917fd28"


def _privileged_states(obs: np.ndarray) -> np.ndarray:
    """Repeat the flattened joint observation as each agent's training state."""
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.repeat(flat[None], obs.shape[0], axis=0)


def _position_slice(env_name: str, obs_dim: int) -> tuple[int, int] | None:
    """Observation slice fed to the released positional embedding.

    `pos_embed: True` in every published config, over an environment-specific slice
    (`obs_pos_embed_start/end`). The built-in navigation observation leads with the
    agent's own position.
    """
    if env_name == "navigation" and obs_dim >= 2:
        return (0, 2)
    return None


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def train(
    *,
    env: str = "navigation",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    mode: str = "implicit",
    hidden_dim: int = 128,
    latent_dim: int = 32,
    n_encoder_layers: int = 1,
    scheduler_heads: int = 1,
    communication_heads: int = 4,
    communication_hops: int = 4,
    negative_slope: float = 1.2,
    gumbel_temperature: float = 0.1,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    world_coef: float = 0.05,
    interaction_coef: float = 0.05,
    rollout_episodes: int = 128,
    ppo_epochs: int = 15,
    num_minibatches: int = 1,
    chunk_length: int = 10,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    """Run the paper/release IWoL learner on a deterministic-seed task."""
    if episodes < 0 or rollout_episodes < 1 or evaluation_episodes < 1:
        raise ValueError("episodes must be non-negative and rollout/evaluation counts positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    environment = make_env(env, n_agents, horizon, seed)
    iwol_mode = IWoLMode(mode)
    position_slice = _position_slice(env, environment.obs_dim)
    learner = IWoLAgent(
        environment.n_agents,
        environment.obs_dim,
        environment.num_actions,
        state_dim=environment.n_agents * environment.obs_dim,
        mode=iwol_mode,
        position_slice=position_slice,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_encoder_layers=n_encoder_layers,
        scheduler_heads=scheduler_heads,
        communication_heads=communication_heads,
        communication_hops=communication_hops,
        negative_slope=negative_slope,
        gumbel_temperature=gumbel_temperature,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_eps,
        entropy_coef=entropy_coef,
        world_coef=world_coef,
        interaction_coef=interaction_coef,
        chunk_length=chunk_length,
    ).to(dev)
    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
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
    actor_rates: list[float] = []
    critic_rates: list[float] = []
    episode_count = 0
    physical_graph = torch.ones(
        environment.n_agents, environment.n_agents, device=dev,
    )
    while episode_count < episodes:
        rollout = IWoLRollout(learner)
        count = min(rollout_episodes, episodes - episode_count)
        for _ in range(count):
            obs, _ = environment.reset(seed=seed + episode_count)
            actor_hidden, critic_hidden = learner.initial_state(dev)
            masks = torch.ones(environment.n_agents, device=dev)
            episode_return = 0.0
            terminated = False
            for _ in range(environment.horizon):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                states = torch.as_tensor(_privileged_states(obs), device=dev)
                previous_actor, previous_critic = actor_hidden, critic_hidden
                step = learner.act(
                    obs_tensor,
                    actor_hidden,
                    critic_hidden,
                    masks,
                    physical_graph,
                )
                actor_hidden, critic_hidden = step.actor_hidden, step.critic_hidden
                next_obs, reward, terminated, truncated, _ = environment.step(
                    step.actions.cpu().numpy(),
                )
                rollout.add(
                    obs=obs_tensor,
                    states=states,
                    physical_graph=physical_graph,
                    step=step,
                    actor_hidden=previous_actor,
                    critic_hidden=previous_critic,
                    masks=masks,
                    team_reward=reward,
                )
                actor_rates.append(learner.graph_rate(step.actor_graph))
                critic_rates.append(learner.graph_rate(step.critic_graph))
                obs = next_obs
                episode_return += reward
                if terminated or truncated:
                    break
            final_mask = torch.zeros(environment.n_agents, device=dev) if terminated else masks
            if terminated:
                bootstrap = torch.zeros(environment.n_agents, device=dev)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                bootstrap, _ = learner.values(
                    obs_tensor, critic_hidden, masks, physical_graph,
                )
            rollout.finish_episode(bootstrap, final_mask)
            returns.append(float(episode_return))
            episode_count += 1
        learner.update(
            rollout.batch(), epochs=ppo_epochs, num_minibatches=num_minibatches,
        )

    configuration = {
        "mode": iwol_mode.value,
        "n_agents": environment.n_agents,
        "horizon": environment.horizon,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "position_slice": list(position_slice) if position_slice else None,
        "n_encoder_layers": n_encoder_layers,
        "scheduler_heads": scheduler_heads,
        "communication_heads": communication_heads,
        "communication_hops": communication_hops,
        "negative_slope": negative_slope,
        "gumbel_temperature": gumbel_temperature,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_epsilon": clip_eps,
        "entropy_coef": entropy_coef,
        "world_coef": world_coef,
        "interaction_coef": interaction_coef,
        "rollout_episodes": rollout_episodes,
        "ppo_epochs": ppo_epochs,
        "num_minibatches": num_minibatches,
        "chunk_length": chunk_length,
        "value_normalization": True,
        "huber_delta": 10.0,
        "max_grad_norm": 10.0,
        "physical_graph": "fully_connected",
    }
    if checkpoint is not None:
        torch.save(
            {
                "model": learner.state_dict(),
                "actor_optimizer": learner.actor_optimizer.state_dict(),
                "critic_optimizer": learner.critic_optimizer.state_dict(),
                "episodes": episode_count,
                "config": configuration,
                "source_revision": SOURCE_REVISION,
            },
            checkpoint,
        )
    final_evaluation = _evaluate(
        learner, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
    )
    random_return = _mean(random_evaluation["returns"])
    return {
        "algorithm": "iwol",
        "variant": "Im-IWoL" if iwol_mode is IWoLMode.IMPLICIT else "Ex-IWoL",
        "source_revision": SOURCE_REVISION,
        "env": env,
        "episodes": episode_count,
        "n_agents": environment.n_agents,
        "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "communication_rate": _mean(final_evaluation["communication_rates"]),
        "training_actor_communication_rate": _mean(actor_rates),
        "training_critic_communication_rate": _mean(critic_rates),
        "checkpoint": checkpoint,
        "config": configuration,
        "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + evaluation_episodes)),
        "initial_evaluation": initial_evaluation,
        "random_evaluation": random_evaluation,
        "final_evaluation": final_evaluation,
        "validation_criterion": {
            "minimum_final_mean_return": random_return + 0.2 * abs(random_return),
            "scope": "every confirmation seed",
        },
    }


@torch.no_grad()
def _evaluate(
    learner: IWoLAgent,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    random_policy: bool = False,
) -> dict[str, list[float]]:
    """Evaluate actor-only execution and report its actual link utilization."""
    environment = make_env(env_name, n_agents, horizon, seed)
    returns, successes, distances, rates = [], [], [], []
    random_generator = np.random.default_rng(seed)
    physical_graph = torch.ones(
        environment.n_agents, environment.n_agents, device=device,
    )
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        actor_hidden, _ = learner.initial_state(device)
        masks = torch.ones(environment.n_agents, device=device)
        episode_return, info = 0.0, {}
        step_rates: list[float] = []
        for _ in range(environment.horizon):
            if random_policy:
                actions = random_generator.integers(
                    environment.num_actions, size=environment.n_agents,
                )
                step_rates.append(0.0)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                output = learner.actor(
                    obs_tensor.unsqueeze(0),
                    actor_hidden.unsqueeze(0),
                    masks.unsqueeze(0),
                    physical_graph.unsqueeze(0),
                    None,
                )
                actions = learner.actor.action_head.sample(
                    output.distribution, deterministic=True,
                )[0].cpu().numpy()
                actor_hidden = output.hidden[0]
                step_rates.append(learner.graph_rate(output.graph))
            obs, reward, terminated, truncated, info = environment.step(actions)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
        rates.append(_mean(step_rates))
    return {
        "returns": returns,
        "successes": successes,
        "mean_distances": distances,
        "communication_rates": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Interactive World Latent.")
    parser.add_argument("--env", default="navigation")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=[mode.value for mode in IWoLMode], default="implicit")
    parser.add_argument("--checkpoint", default="iwol.pt")
    args = parser.parse_args()
    train(
        env=args.env,
        episodes=args.episodes,
        n_agents=args.n_agents,
        seed=args.seed,
        mode=args.mode,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
