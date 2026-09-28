"""Train MASIA on a cooperative task and save a checkpoint.

Run:
    python examples/train_masia.py --episodes 500

QMIX's episode-replay recurrent sibling with self-supervised message aggregation:
whole episodes are collected with TWO hidden states threaded through time — the
Q-network GRU and the aggregation encoder's integration GRU — and replayed in
batches from zeros. The update unrolls online and target streams (one target
encoder feeds both the TD-target Q input and the SPR projection targets), mixes
chosen-action Qs with the QMIX mixer under a double-Q target, and adds the paper's
representation losses: global-state reconstruction from z (Eq. 3), K-step SPR
latent prediction through the residual transition model (Eq. 4-7, plus the
official k=0 alignment term), and reward prediction from the rolled-out latents.
The TD gradient deliberately reaches the encoder; one Adam step covers all online
parameters. `train()` is importable; built from the public modMARL API.

Discount 0.99, one update per episode, and Adam(lr) all match the released
`masia.yaml`/`default.yaml`. Reward standardization follows the release's
`RunningMeanStd`: statistics over the padded tensor, unbiased variance, no epsilon.
The reward-prediction auxiliary regresses only latents that have a real successor
reward; the release additionally predicts the padded final step, a buffer artifact
with no target, so that one term is dropped here.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from marl_envs import make_env
from modmarl.algorithms.masia import MASIAAgent
from modmarl.common.replay import EpisodeReplayBuffer

GRAD_CLIP = 10.0


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 500,
    seed: int = 7,
    lr: float = 5e-4,
    hidden_dim: int = 64,
    enc_hidden_dim: int = 32,
    z_slot_dim: int = 8,
    spr_dim: int = 32,
    mixer_hidden_dim: int = 32,
    buffer_episodes: int = 5000,
    batch_episodes: int = 32,
    warmup_episodes: int = 32,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    gamma: float = 0.99,
    repr_coef: float = 1.0,
    spr_coef: float = 1.0,
    rew_pred_coef: float = 1.0,
    pred_len: int = 2,
    updates_per_episode: int = 1,
    target_update_every: int = 200,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    if env == "ndq_hallway":
        environment.reward_win = 1.0
    n, obs_dim, num_actions = environment.n_agents, environment.obs_dim, environment.num_actions
    agent = MASIAAgent(
        n_agents=n, obs_dim=obs_dim, action_dim=num_actions, hidden_dim=hidden_dim,
        enc_hidden_dim=enc_hidden_dim, z_slot_dim=z_slot_dim, spr_dim=spr_dim,
        mixer_hidden_dim=mixer_hidden_dim,
    ).to(dev)
    opt = torch.optim.Adam(_online_params(agent), lr=lr)
    replay = EpisodeReplayBuffer(buffer_episodes, environment.horizon, n, obs_dim)
    reward_stats = _RunningRewardStats(dev)

    evaluation_seed = seed + 100_000
    initial = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)
    random = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        random_policy=True,
    )
    returns: list[float] = []
    env_steps = 0
    for episode in range(episodes):
        frac = min(1.0, env_steps / 50_000)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        episode_return, episode_steps = _collect_episode(
            agent, environment, replay, epsilon, num_actions, seed + episode, dev,
        )
        returns.append(episode_return)
        env_steps += episode_steps

        if len(replay) >= max(batch_episodes, warmup_episodes):
            for _ in range(updates_per_episode):
                _update(
                    agent, opt, replay.sample(batch_episodes, dev), gamma,
                    repr_coef, spr_coef, rew_pred_coef, pred_len, reward_stats,
                )
        if episode % target_update_every == 0 and episode > 0:
            agent.update_targets()
        if episode % 20 == 0:
            print(f"episode {episode:4d}  eps {epsilon:4.2f}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(agent.state_dict(), checkpoint)
    final = _evaluate(agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev)

    return {
        "algorithm": "masia",
        "env": env,
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
        "communication_rate": 1.0,
        "validation_criterion": {"minimum_win_rate": 0.8, "scope": "every fixed seed"},
    }


def _online_params(agent) -> list[torch.nn.Parameter]:
    """All online parameters — one Adam over agent, encoder, SPR heads, model and mixer."""
    return (
        list(agent.encoder.parameters())
        + list(agent.q_network.parameters())
        + list(agent.projection.parameters())
        + list(agent.predictor.parameters())
        + list(agent.transition_model.parameters())
        + list(agent.mixer.parameters())
    )


@torch.no_grad()
def _collect_episode(agent, environment, replay, epsilon, num_actions, episode_seed, dev):
    obs, _ = environment.reset(seed=episode_seed)
    q_hidden, enc_hidden = agent.init_state(1, dev)
    n = environment.n_agents
    obs_list, action_list, reward_list, done_list = [obs], [], [], []
    episode_return = 0.0
    steps = 0
    for _ in range(environment.horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        q, q_hidden, enc_hidden = agent(obs_t, q_hidden, enc_hidden)
        greedy = q.argmax(dim=-1).squeeze(0).cpu().numpy()
        explore = np.random.random(n) < epsilon
        action = np.where(explore, np.random.randint(0, num_actions, size=n), greedy).astype(np.int64)

        next_obs, reward, terminated, truncated, _ = environment.step(action)
        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(float(terminated))
        obs = next_obs
        episode_return += reward
        steps += 1
        if terminated or truncated:
            break

    replay.add_episode(
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(action_list, dtype=np.int64),
        rewards=np.asarray(reward_list, dtype=np.float32),
        dones=np.asarray(done_list, dtype=np.float32),
    )
    return episode_return, steps


@torch.no_grad()
def _evaluate(agent, env_name, n_agents, horizon, seed, episodes, device, *, random_policy=False):
    environment = make_env(env_name, n_agents, horizon, seed)
    if env_name == "ndq_hallway":
        environment.reward_win = 1.0
    generator = np.random.default_rng(seed)
    returns, successes = [], []
    was_training = agent.training
    agent.eval()
    try:
        for episode in range(episodes):
            obs, _ = environment.reset(seed=seed + episode)
            q_hidden, enc_hidden = agent.init_state(1, device)
            episode_return, info = 0.0, {}
            for _ in range(environment.horizon):
                if random_policy:
                    action = generator.integers(environment.num_actions, size=environment.n_agents)
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values, q_hidden, enc_hidden = agent(obs_tensor, q_hidden, enc_hidden)
                    action = q_values.argmax(-1).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            successes.append(float(bool(info.get("success", False))))
    finally:
        environment.close()
        agent.train(was_training)
    return {
        "returns": returns,
        "successes": successes,
        "win_rate": float(np.mean(successes)) if successes else 0.0,
    }


class _RunningRewardStats:
    """Released running mean/variance update over valid replay rewards."""

    def __init__(self, device: torch.device) -> None:
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = 1e-4

    def standardize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Released ``RunningMeanStd`` semantics: padded steps included, unbiased
        variance, and no epsilon in the denominator."""
        values = rewards.reshape(-1)
        count = values.numel()
        mean, variance = values.mean(), values.var(unbiased=True)
        delta = mean - self.mean
        total = self.count + count
        self.mean = self.mean + delta * count / total
        second = (
            self.var * self.count + variance * count
            + delta.square() * self.count * count / total
        )
        self.var, self.count = second / total, total
        return (rewards - self.mean) / torch.sqrt(self.var)


def _update(
    agent, opt, batch, gamma, repr_coef, spr_coef, rew_pred_coef, pred_len,
    reward_stats: _RunningRewardStats | None = None,
) -> None:
    batch_size, horizon = batch.actions.shape[:2]
    n = agent.n_agents
    dev = batch.obs.device

    # Unroll both online streams (encoder GRU + Q GRU) and their target twins over the
    # episode from zero hiddens; the single target encoder serves the TD-target Q input
    # and the SPR projection targets.
    q_hidden, enc_hidden = agent.init_state(batch_size, dev)
    target_q_hidden, target_enc_hidden = agent.init_state(batch_size, dev)
    _, momentum_enc_hidden = agent.init_state(batch_size, dev)
    online_q, online_z, target_q, target_proj = [], [], [], []
    for t in range(horizon + 1):
        obs_t = batch.obs[:, t]
        z, enc_hidden = agent.enc_forward(obs_t, enc_hidden)
        q, q_hidden = agent.q_forward(obs_t, z, q_hidden)
        online_q.append(q)
        online_z.append(z)
        proj, _, momentum_enc_hidden = agent.target_project_enc(obs_t, momentum_enc_hidden)
        with torch.no_grad():
            target_z, target_enc_hidden = agent.target_enc_forward(obs_t, target_enc_hidden)
        with torch.no_grad():
            q_t, target_q_hidden = agent.q_forward(obs_t, target_z, target_q_hidden, target=True)
        target_q.append(q_t)
        target_proj.append(proj)
    online_q = torch.stack(online_q, dim=1)                     # (B, T+1, n, A)
    online_z = torch.stack(online_z, dim=1)                     # (B, T+1, n*z_slot)
    target_q = torch.stack(target_q, dim=1)
    target_proj = torch.stack(target_proj, dim=1)               # (B, T+1, spr_dim)

    # TD loss: chosen Qs mixed by the online mixer; double-Q target mixed by the target
    # mixer. z is NOT detached — the TD gradient reaches the encoder (rl_signal).
    chosen = online_q[:, :-1].gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)    # (B, T, n)
    state = batch.obs.reshape(batch_size, horizon + 1, -1)
    q_tot = agent.mixer(chosen.reshape(-1, n), state[:, :-1].reshape(-1, state.shape[-1]))
    q_tot = q_tot.view(batch_size, horizon)
    with torch.no_grad():
        next_argmax = online_q[:, 1:].argmax(dim=-1, keepdim=True)                   # double Q
        next_target = target_q[:, 1:].gather(-1, next_argmax).squeeze(-1)
        next_tot = agent.target_mixer(next_target.reshape(-1, n), state[:, 1:].reshape(-1, state.shape[-1]))
        next_tot = next_tot.view(batch_size, horizon)
        rewards = (
            reward_stats.standardize(batch.rewards)
            if reward_stats is not None else batch.rewards
        )
        y = rewards + gamma * (1.0 - batch.dones) * next_tot
    td_error = (q_tot - y) * batch.mask
    td_loss = (td_error ** 2).sum() / batch.mask.sum()

    # Reconstruction (paper Eq. 3): decode z against the global state (concat obs) over
    # every real observation step — one more than the TD mask, because an episode's
    # final observation is real even though no action follows it. The observation that
    # would follow a terminal transition does not exist, so it is dropped, reproducing
    # the release's `mask[:, 1:] *= (1 - terminated[:, :-1])`.
    obs_mask = torch.cat(
        [torch.ones_like(batch.mask[:, :1]), batch.mask * (1.0 - batch.dones)], dim=1,
    )    # (B, T+1)
    recon_error = ((agent.encoder.decode(online_z) - state) ** 2).mean(dim=-1)
    recon_loss = (recon_error * obs_mask).sum() / obs_mask.sum()

    # SPR (paper Eq. 4-7 plus the official k=0 alignment term): roll the residual
    # transition model K steps in latent space; predictions go through projection +
    # predictor, targets came from the frozen target encoder + projection above.
    # Level k pairs the prediction from base step t with step t+k, so targets and
    # masks shift by k. The reward head regresses r_{t+k} from the same latents.
    actions_onehot = F.one_hot(batch.actions, online_q.shape[-1]).to(online_z.dtype)
    spr_loss, reward_loss = 0.0, 0.0
    rollout = online_z
    for k in range(pred_len + 1):
        if k > 0:
            # Advance level k-1 (dropping its last step) with actions a_{k-1..T-1}.
            rollout = agent.transition_model(rollout[:, :-1], actions_onehot[:, k - 1:])
        # clamp_min guards episodes shorter than the rollout depth: a level with no
        # valid steps contributes zero instead of 0/0.
        spr_error = ((agent.project(rollout) - target_proj[:, k:]) ** 2).sum(dim=-1)
        spr_loss = spr_loss + (spr_error * obs_mask[:, k:]).sum() / obs_mask[:, k:].sum().clamp_min(1.0)
        predicted_r = agent.transition_model.predict_reward(rollout[:, :-1]).squeeze(-1)
        r_error = (predicted_r - batch.rewards[:, k:]) ** 2
        reward_loss = reward_loss + (r_error * batch.mask[:, k:]).sum() / batch.mask[:, k:].sum().clamp_min(1.0)

    loss = td_loss + repr_coef * (recon_loss + spr_coef * spr_loss + rew_pred_coef * reward_loss)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(_online_params(agent), GRAD_CLIP)
    opt.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MASIA on a cooperative task.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="masia.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
