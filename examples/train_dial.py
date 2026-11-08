"""Train DIAL or RIAL (Foerster et al., NeurIPS 2016) on the switch riddle.

Defaults reproduce the paper's switch setup (Section 6; release script
``run_switch_3-dial.sh``): n = 3 prisoners, T = 4n - 6, batches of 32 parallel
episodes, epsilon = 0.05, gamma = 1, sigma = 2, learning rate 5e-4, a 1-bit
channel heard only by the next visitor, and target copies every 100 batches.
The paper reports all four switch variants solving n = 3 within 5k batches
(160k episodes). ``episodes`` counts individual episodes; each training
iteration consumes ``batch_size`` of them.

The runner only steps environments and assembles tensors. Networks, the channel,
action selection, the loss and target updates live in
``modmarl.algorithms.dial``. Any ``marl_envs`` task also works; tasks without the
switch riddle's per-step queries use full availability and broadcast channels.
The switch riddle is built directly from ``marl_envs.switch`` rather than through
``make_env``, so adding it left the factory used by frozen confirmations unchanged.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from marl_envs import make_env
from marl_envs.switch import SwitchRiddleEnv
from modmarl.algorithms.dial import DIALAgent, DIALConfig


def _make(env: str, n_agents: int, horizon: int, seed: int):
    """Build the switch riddle directly (paper horizon 4n - 6); other tasks via ``make_env``."""
    if env == "switch":
        return SwitchRiddleEnv(n_agents=n_agents, seed=seed)
    return make_env(env, n_agents, horizon, seed)


def _queries(env, n_agents: int, action_dim: int):
    available = getattr(env, "available_actions", None)
    mask = getattr(env, "communication_mask", None)
    senders = getattr(env, "message_senders", None)
    return (
        available() if callable(available) else np.ones((n_agents, action_dim), np.float32),
        mask() if callable(mask) else None,
        senders() if callable(senders) else None,
    )


def _stack_queries(envs, n_agents: int, action_dim: int, device: torch.device):
    available, masks, senders = zip(*(_queries(env, n_agents, action_dim) for env in envs), strict=True)
    def tensor(items):
        if any(item is None for item in items):
            return None
        return torch.as_tensor(np.stack(items), dtype=torch.float32, device=device)
    return tensor(available), tensor(masks), tensor(senders)


def run_batch(learner: DIALAgent, envs, seeds, device: torch.device, *, train: bool,
              silence: bool = False) -> dict:
    """Run one batch of parallel episodes; update the learner when ``train``.

    ``silence`` delivers no messages (an all-zero channel mask), the message
    ablation used to check that a learned policy relies on communication.
    """
    learner.train(train)
    n_agents, action_dim = learner.n_agents, learner.action_dim
    obs = np.stack([env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds, strict=True)])
    learner.begin_episode(len(envs), device)
    running = np.ones(len(envs), dtype=bool)
    returns = np.zeros(len(envs), dtype=np.float64)
    successes = np.zeros(len(envs), dtype=bool)
    horizon = max(env.horizon for env in envs)
    for _ in range(horizon):
        if not running.any():
            break
        available, mask, senders = _stack_queries(envs, n_agents, action_dim, device)
        if silence:
            mask = torch.zeros(len(envs), n_agents, n_agents, device=device)
        with torch.set_grad_enabled(train):
            actions = learner.act(
                torch.as_tensor(obs, dtype=torch.float32, device=device), available,
                communication_mask=mask, message_senders=senders,
            ).cpu().numpy()
        rewards = np.zeros(len(envs), dtype=np.float32)
        terminated = np.zeros(len(envs), dtype=np.float32)
        active = running.astype(np.float32)
        for index, env in enumerate(envs):
            if not running[index]:
                continue
            next_obs, reward, term, trunc, info = env.step(actions[index])
            rewards[index], terminated[index] = reward, float(term)
            returns[index] += reward
            obs[index] = next_obs
            if term or trunc:
                running[index] = False
                successes[index] = bool(info.get("success", False))
        learner.observe(rewards, terminated, active)
    metrics = {"returns": returns.tolist(), "successes": successes.tolist()}
    if train:
        final_available, final_mask, final_senders = _stack_queries(envs, n_agents, action_dim, device)
        metrics.update(learner.finish_episode(
            torch.as_tensor(obs, dtype=torch.float32, device=device), final_available,
            final_communication_mask=final_mask, final_message_senders=final_senders,
        ))
    return metrics


def _oracle_rate(env, seeds) -> float:
    """Mean return of a full-information team: +1 whenever everyone visits in time."""
    rates = []
    for seed in seeds:
        env.reset(seed=int(seed))
        rates.append(float(env.everyone_has_visited()))
    return float(np.mean(rates))


def train(
    *,
    env: str = "switch", mode: str = "dial", n_agents: int = 3, horizon: int = 25,
    episodes: int = 32 * 200, batch_size: int = 32, seed: int = 7,
    message_bits: int = 1, noise_std: float = 2.0, parameter_sharing: bool = True,
    learning_rate: float = 5e-4, gamma: float = 1.0, epsilon: float = 0.05,
    target_update_interval: int = 100, hidden_dim: int = 128,
    evaluation_episodes: int = 256, evaluation_interval: int = 0,
    device: str = "cpu", checkpoint: str | None = None, log_interval: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    envs = [_make(env, n_agents, horizon, seed + index) for index in range(batch_size)]
    sample = envs[0]
    aggregation = "sum" if hasattr(sample, "communication_mask") else "concat"
    config = DIALConfig(
        mode=mode, message_bits=message_bits, noise_std=noise_std,
        parameter_sharing=parameter_sharing, message_aggregation=aggregation,
        learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
        target_update_interval=target_update_interval, hidden_dim=hidden_dim,
    )
    learner = DIALAgent(sample.n_agents, sample.obs_dim, sample.num_actions, config).to(dev)

    evaluation_seed = seed + 1_000_000
    evaluation_seeds = np.arange(evaluation_seed, evaluation_seed + evaluation_episodes)
    evaluation_envs = [_make(env, n_agents, horizon, int(s)) for s in evaluation_seeds[:batch_size]]

    def evaluate(silence: bool = False) -> dict:
        returns, successes = [], []
        for start in range(0, evaluation_episodes, batch_size):
            chunk = evaluation_seeds[start:start + batch_size]
            result = run_batch(learner, evaluation_envs[: len(chunk)], chunk, dev, train=False,
                               silence=silence)
            returns += result["returns"]
            successes += result["successes"]
        return {"returns": returns, "successes": [float(s) for s in successes],
                "mean_return": float(np.mean(returns))}

    initial_evaluation = evaluate()
    iterations = math.ceil(episodes / batch_size)
    # Training episodes draw their seeds from the run seed, so runs with nearby seeds
    # see independent task draws; evaluation seeds stay shared for paired comparison.
    episode_seeds = np.random.default_rng([seed, 10_000])
    returns: list[float] = []
    curve: list[dict] = []
    for iteration in range(iterations):
        seeds = episode_seeds.integers(0, 2**31 - 1, size=batch_size)
        result = run_batch(learner, envs, seeds, dev, train=True)
        returns.append(float(np.mean(result["returns"])))
        if evaluation_interval and (iteration + 1) % evaluation_interval == 0:
            curve.append({"iteration": iteration + 1, "mean_return": evaluate()["mean_return"]})
        if log_interval and (iteration + 1) % log_interval == 0:
            print(f"iteration {iteration + 1:5d}  train return {returns[-1]: .3f}  "
                  f"loss {result['loss']:.4f}")

    final_evaluation = evaluate()
    silenced_evaluation = evaluate(silence=True)
    summary = {
        "algorithm": "dial" if mode == "dial" else "rial", "mode": mode, "env": env,
        "episodes": iterations * batch_size, "iterations": iterations,
        "n_agents": sample.n_agents, "returns": returns,
        "final_return": returns[-1] if returns else 0.0,
        "best_return": max(returns) if returns else 0.0, "checkpoint": checkpoint,
        "evaluation_seeds": evaluation_seeds.tolist(),
        "initial_evaluation": initial_evaluation, "final_evaluation": final_evaluation,
        "silenced_evaluation": silenced_evaluation,
        "evaluation_curve": curve,
    }
    if hasattr(sample, "everyone_has_visited"):
        oracle = _oracle_rate(_make(env, n_agents, horizon, 0), evaluation_seeds)
        summary["oracle_mean_return"] = oracle
        summary["normalised_final_return"] = (
            final_evaluation["mean_return"] / oracle if oracle > 0 else float("nan")
        )
    if checkpoint is not None:
        torch.save({"model": learner.state_dict(), "optimizer": learner.optimizer.state_dict(),
                    "config": config, "iterations": iterations}, checkpoint)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DIAL or RIAL on the switch riddle.")
    parser.add_argument("--mode", choices=("dial", "rial"), default="dial")
    parser.add_argument("--env", default="switch")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=25, help="ignored by the switch riddle")
    parser.add_argument("--episodes", type=int, default=32 * 200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    summary = train(mode=args.mode, env=args.env, n_agents=args.n_agents, horizon=args.horizon,
                    episodes=args.episodes,
                    seed=args.seed, checkpoint=args.checkpoint, log_interval=20)
    print(f"final greedy return {summary['final_evaluation']['mean_return']:.3f}")


if __name__ == "__main__":
    main()
