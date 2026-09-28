"""Train SMS on a cooperative task and save a checkpoint.

Run:
    python examples/train_sms.py --episodes 500

DOP-style actor-critic on the episode-replay recurrent scaffold, following the released
two-stream training (paper Eq. 8-9). Whole episodes are collected with the previous
action threaded into the controller input and Gaussian message noise injected, the policy
sampling its softmax under full communication (no dropout at rollout) until t_selector env
steps and through the sign-of-selector gate after. Each learner step runs BOTH streams on
their own optimizers:

  * an ON-POLICY update on the latest episodes — the twin dueling critics + mixer on
    TD(lambda) targets under the taken actions, the actor (paper Eq. 10: softmax policies
    pushed through critic1 + mixer, selector off / dropout on) plus the release's
    self-normalised entropy gradient, and the learner-side selector regressed onto noisy,
    k_i(s)-scaled Shapley Message Value labels
    (target policy, online critics' min) over the real steps;
  * an OFF-POLICY update on a uniform replay sample — only the twin critics + mixer, with
    messages RELABELLED under the current target encoder (its softmax under full
    communication) and a one-step (lambda = 0) target.

The policy remains fixed for each released ``batch_size_run=8`` collection group before one
off-policy and one on-policy learner update. Hard target copies and the selector sync into the
acting agent share one cadence.
``train()`` is importable; built from the public modMARL API.

The defaults reproduce the released SMS optimizer, replay, discount, and selector
schedule. Environment-specific controller features remain explicit.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from marl_envs import make_env
from modmarl.algorithms.sms import (
    LinearDecompositionMixer,
    SMSAgent,
    SMSCritic,
    shapley_message_values,
)
from modmarl.common.replay import EpisodeReplayBuffer

GRAD_CLIP = 10.0


def train(
    *,
    env: str = "simple_spread",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 500,
    seed: int = 7,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    selector_lr: float = 1e-4,
    hidden_dim: int = 64,
    msg_dim: int = 8,
    msg_hidden_dim: int = 32,
    selector_hidden_dim: int = 32,
    critic_hidden_dim: int = 128,
    mixer_embed_dim: int = 32,
    use_rnn: bool = False,
    on_buffer_episodes: int = 128,
    off_buffer_episodes: int = 5000,
    batch_episodes: int = 32,
    off_batch_episodes: int = 64,
    warmup_episodes: int = 128,
    gamma: float = 0.99,
    td_lambda: float = 0.6,
    entropy_coef: float = 0.03,
    dropout_p: float = 0.5,
    smv_sample_size: int = 2,
    t_selector: int = 300_000,
    collection_batch_size: int = 8,
    updates_per_collection: int = 1,
    target_update_every: int = 200,
    evaluation_episodes: int = 100,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    environment = make_env(env, n_agents, horizon, seed)
    n, obs_dim, num_actions = environment.n_agents, environment.obs_dim, environment.num_actions
    state_dim = n * obs_dim
    agent = SMSAgent(
        n_agents=n, obs_dim=obs_dim, action_dim=num_actions, hidden_dim=hidden_dim,
        msg_dim=msg_dim, msg_hidden_dim=msg_hidden_dim, selector_hidden_dim=selector_hidden_dim,
        use_rnn=use_rnn,
    ).to(dev)
    critic1 = SMSCritic(n, obs_dim, state_dim, num_actions, critic_hidden_dim).to(dev)
    critic2 = SMSCritic(n, obs_dim, state_dim, num_actions, critic_hidden_dim).to(dev)
    mixer = LinearDecompositionMixer(n, state_dim, mixer_embed_dim).to(dev)
    target_agent = copy.deepcopy(agent)
    target_critic1 = copy.deepcopy(critic1)
    target_critic2 = copy.deepcopy(critic2)
    target_mixer = copy.deepcopy(mixer)
    learner_selector = copy.deepcopy(agent.selector)
    actor_opt = torch.optim.Adam(_actor_params(agent), lr=actor_lr)
    critic_opt = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()) + list(mixer.parameters()), lr=critic_lr,
    )
    selector_opt = torch.optim.Adam(learner_selector.parameters(), lr=selector_lr)
    on_replay = EpisodeReplayBuffer(on_buffer_episodes, environment.horizon, n, obs_dim)
    off_replay = EpisodeReplayBuffer(off_buffer_episodes, environment.horizon, n, obs_dim)

    modules = (agent, target_agent, critic1, critic2, target_critic1, target_critic2,
               mixer, target_mixer, learner_selector)
    optimizers = (actor_opt, critic_opt, selector_opt)
    hyper = dict(gamma=gamma, td_lambda=td_lambda, entropy_coef=entropy_coef,
                 dropout_p=dropout_p, smv_sample_size=smv_sample_size)

    evaluation_seed = seed + 100_000
    initial_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        selector_on=False,
    )
    random_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        selector_on=False, random_policy=True,
    )

    returns: list[float] = []
    t_env = 0
    last_target_episode = 0
    for episode in range(episodes):
        episode_return, t_env = _collect_episode(
            agent, environment, (on_replay, off_replay), t_env, t_selector, seed + episode, dev,
        )
        returns.append(episode_return)

        episodes_collected = episode + 1
        collection_complete = episodes_collected % collection_batch_size == 0
        learner_ready = (
            len(on_replay) >= max(batch_episodes, warmup_episodes)
            and len(off_replay) >= off_batch_episodes
        )
        if collection_complete and learner_ready:
            for _ in range(updates_per_collection):
                # Released order: the off-policy stream is updated before the on-policy
                # one (`on_off_run.py`), and relabelling sees the live selector gate.
                _update(*modules, *optimizers, off_replay.sample(off_batch_episodes, dev),
                        **hyper, off=True, selector_on=t_env >= t_selector)
                _update(*modules, *optimizers, on_replay.sample_latest(batch_episodes, dev),
                        **hyper, off=False)
            release_episode = episodes_collected - collection_batch_size
            if release_episode - last_target_episode >= target_update_every:
                agent.sync_selector(learner_selector)
                target_agent.load_state_dict(agent.state_dict())
                target_critic1.load_state_dict(critic1.state_dict())
                target_critic2.load_state_dict(critic2.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
                last_target_episode = release_episode
        if episode % 20 == 0:
            print(f"episode {episode:4d}  return {episode_return:7.2f}", flush=True)

    if checkpoint is not None:
        torch.save(
            {"agent": agent.state_dict(), "critic1": critic1.state_dict(),
             "critic2": critic2.state_dict(), "mixer": mixer.state_dict(),
             "selector": learner_selector.state_dict()},
            checkpoint,
        )

    final_evaluation = _evaluate(
        agent, env, n_agents, horizon, evaluation_seed, evaluation_episodes, dev,
        selector_on=t_env >= t_selector,
    )
    return {
        "algorithm": "sms",
        "env": env,
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
        "communication_rate": final_evaluation["communication_rate"],
        "config": {
            "env": env,
            "episodes": episodes,
            "seed": seed,
            "n_agents": n,
            "horizon": environment.horizon,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "selector_lr": selector_lr,
            "hidden_dim": hidden_dim,
            "msg_dim": msg_dim,
            "msg_hidden_dim": msg_hidden_dim,
            "selector_hidden_dim": selector_hidden_dim,
            "critic_hidden_dim": critic_hidden_dim,
            "mixer_embed_dim": mixer_embed_dim,
            "use_rnn": use_rnn,
            "on_buffer_episodes": on_buffer_episodes,
            "off_buffer_episodes": off_buffer_episodes,
            "batch_episodes": batch_episodes,
            "off_batch_episodes": off_batch_episodes,
            "warmup_episodes": warmup_episodes,
            "gamma": gamma,
            "td_lambda": td_lambda,
            "entropy_coef": entropy_coef,
            "dropout_p": dropout_p,
            "smv_sample_size": smv_sample_size,
            "t_selector": t_selector,
            "collection_batch_size": collection_batch_size,
            "updates_per_collection": updates_per_collection,
            "target_update_every": target_update_every,
        },
        "replay_capacities": {"on_policy": on_replay.capacity, "off_policy": off_replay.capacity},
        "checkpoint": checkpoint,
    }


@torch.no_grad()
def _evaluate(
    agent: SMSAgent,
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    episodes: int,
    device: torch.device,
    *,
    selector_on: bool,
    random_policy: bool = False,
) -> dict[str, list[float] | float]:
    """Evaluate the released controller without rollout message noise or dropout."""
    environment = make_env(env_name, n_agents, horizon, seed)
    generator = np.random.default_rng(seed)
    returns, successes, distances = [], [], []
    active_links = possible_links = 0.0
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        hidden = agent.init_state(1, device)
        last_action = torch.zeros(
            1, environment.n_agents, agent.action_dim, device=device,
        )
        episode_return, info = 0.0, {}
        for _ in range(environment.horizon):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = agent.comm_mask(
                obs_tensor, last_action, selector_on=selector_on, dropout_p=0.0,
            )
            logits, hidden = agent(
                obs_tensor, hidden, mask, last_action, noise=False,
            )
            action = (
                generator.integers(
                    environment.num_actions, size=environment.n_agents,
                )
                if random_policy
                else logits.squeeze(0).argmax(dim=-1).cpu().numpy()
            )
            last_action = F.one_hot(
                torch.as_tensor(action, device=device), agent.action_dim,
            ).to(torch.float32).unsqueeze(0)
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
            active_links += float(mask.sum().item())
            possible_links += environment.n_agents * (environment.n_agents - 1)
            if terminated or truncated:
                break
        returns.append(float(episode_return))
        successes.append(float(bool(info.get("success", False))))
        distances.append(float(info.get("mean_distance", float("nan"))))
    environment.close()
    return {
        "returns": returns,
        "successes": successes,
        "mean_distances": distances,
        "communication_rate": active_links / max(1.0, possible_links),
    }


def _actor_params(agent) -> list[torch.nn.Parameter]:
    """Policy parameters — trunk, message encoder and head. The acting selector is synced
    from the learner-side copy, never trained by gradient."""
    return (
        list(agent.fc1.parameters())
        + list(agent.gru.parameters())
        + list(agent.msg_encoder.parameters())
        + list(agent.head.parameters())
    )


def _last_actions(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Previous-action one-hots for the controller input: (B, T, n) -> (B, T+1, n, A),
    with a zero vector at t = 0 and onehot(actions[t-1]) at every later step."""
    onehot = F.one_hot(actions, num_actions).to(torch.float32)
    return torch.cat([torch.zeros_like(onehot[:, :1]), onehot], dim=1)


@torch.no_grad()
def _collect_episode(agent, environment, replays, t_env, t_selector, episode_seed, dev) -> tuple[float, int]:
    obs, _ = environment.reset(seed=episode_seed)
    hidden = agent.init_state(1, dev)
    n = environment.n_agents
    last_action = torch.zeros(1, n, agent.action_dim, device=dev)
    obs_list, action_list, reward_list, done_list = [obs], [], [], []
    episode_return = 0.0
    for _ in range(environment.horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        # Rollouts communicate WITHOUT dropout (as the official controller does) but WITH
        # the released Gaussian message noise; the selector gate switches on at t_selector.
        mask = agent.comm_mask(obs_t, last_action, selector_on=t_env >= t_selector, dropout_p=0.0)
        logits, hidden = agent(obs_t, hidden, mask, last_action, noise=True)
        action = torch.distributions.Categorical(logits=logits.squeeze(0)).sample().cpu().numpy().astype(np.int64)
        last_action = F.one_hot(torch.as_tensor(action, device=dev), agent.action_dim).to(torch.float32).unsqueeze(0)

        next_obs, reward, terminated, truncated, _ = environment.step(action)
        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(float(terminated))
        obs = next_obs
        episode_return += reward
        t_env += 1
        if terminated or truncated:
            break

    episode_data = {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.int64),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "dones": np.asarray(done_list, dtype=np.float32),
    }
    for replay in replays:
        replay.add_episode(**episode_data)
    return episode_return, t_env


def _td_lambda_targets(rewards, dones, mask, values, gamma, td_lambda):
    """Backward lambda-return recursion (PyMARL's build_td_lambda_targets shape).

    ``values`` (B, T+1) holds target-side mixed values — chosen-action (or relabelled)
    values at steps 0..T-1 and the soft bootstrap at the final observation; padding and
    termination are handled by ``mask``/``dones`` zeroing the tail. Returns targets (B, T)."""
    horizon = rewards.shape[1]
    ret = torch.zeros_like(values)
    ret[:, -1] = values[:, -1] * (1.0 - dones.sum(dim=1).clamp(max=1.0))
    for t in range(horizon - 1, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
            rewards[:, t] + (1.0 - td_lambda) * gamma * values[:, t + 1] * (1.0 - dones[:, t])
        )
    return ret[:, :-1]


def _selector_loss(preds, labels):
    """Diagonal-masked selector regression (paper Eq. 11): predictions are zeroed on the
    diagonal and the mean rescaled by n/(n-1) so it runs over the n-1 real senders; the
    label diagonal (which holds Q(empty) by construction) then contributes a constant with
    zero gradient."""
    n = preds.shape[-1]
    off_diagonal = 1.0 - torch.eye(n, device=preds.device)
    return ((preds * off_diagonal - labels) ** 2).mean() * n / (n - 1)


def _normalised_entropy(entropy, mask):
    """Released entropy objective with a finite zero-entropy limit."""
    mean_entropy = (entropy * mask).sum() / mask.sum()
    denominator = mean_entropy.detach().clamp_min(torch.finfo(mean_entropy.dtype).eps)
    return mean_entropy / denominator


def _target_values(agent, target_agent, target_critic1, target_critic2, target_mixer,
                   batch, state, last_actions, taken, off, dev, selector_on=False):
    """Per-step target mixed values (B, T+1). On-policy: taken actions with a softmax
    bootstrap at the final observation. Off-policy: the target policy's relabelled softmax
    at every step (paper Eq. 9).

    The release relabels through ``target_mac.forward(batch, t, t_env=t_env)`` with
    ``msg_mask_agent=None``, so the selector gate is live once ``t_env >= t_selector`` and
    the message noise is injected (``test_mode`` defaults False). Relabelling under full,
    noiseless communication would evaluate a policy the agent never runs."""
    batch_size, horizon = batch.actions.shape[:2]
    n = agent.n_agents
    target_hidden = target_agent.init_state(batch_size, dev)
    pre_hiddens, value_seq = [], []
    full_mask = 1.0 - torch.eye(n, device=dev).expand(batch_size, n, n)
    for t in range(horizon + 1):
        pre_hiddens.append(target_hidden)
        if off:
            mask_t = target_agent.comm_mask(
                batch.obs[:, t], last_actions[:, t], selector_on=selector_on, dropout_p=0.0,
            )
        else:
            mask_t = full_mask
        logits_t, target_hidden = target_agent(
            batch.obs[:, t], target_hidden, mask_t, last_actions[:, t], noise=off,
        )
        if off or t == horizon:
            probs_t = torch.softmax(logits_t, dim=-1)                    # relabel / bootstrap
        else:
            probs_t = taken[:, t]                                        # on-stream taken action
        q_t = torch.min(
            target_critic1(batch.obs[:, t], state[:, t], probs_t),
            target_critic2(batch.obs[:, t], state[:, t], probs_t),
        ).squeeze(-1)                                                    # (B, n)
        value_seq.append(target_mixer(q_t, state[:, t]))                 # (B,)
    return torch.stack(value_seq, dim=1), pre_hiddens


def _update(agent, target_agent, critic1, critic2, target_critic1, target_critic2,
            mixer, target_mixer, learner_selector, actor_opt, critic_opt, selector_opt, batch,
            gamma, td_lambda, entropy_coef, dropout_p, smv_sample_size, *,
            off=False, selector_on=False) -> None:
    batch_size, horizon = batch.actions.shape[:2]
    n = agent.n_agents
    dev = batch.obs.device
    num_actions = critic1.action_dim
    state = batch.obs.reshape(batch_size, horizon + 1, -1)
    last_actions = _last_actions(batch.actions, num_actions)            # (B, T+1, n, A)
    taken = F.one_hot(batch.actions, num_actions).to(batch.obs.dtype)   # (B, T, n, A)
    obs_flat = batch.obs[:, :-1].reshape(batch_size * horizon, n, -1)
    state_flat = state[:, :-1].reshape(batch_size * horizon, -1)
    taken_flat = taken.reshape(batch_size * horizon, n, num_actions)

    with torch.no_grad():
        values, pre_hiddens = _target_values(
            agent, target_agent, target_critic1, target_critic2, target_mixer,
            batch, state, last_actions, taken, off, dev, selector_on=selector_on,
        )
        lambd = 0.0 if off else td_lambda
        targets = _td_lambda_targets(batch.rewards, batch.dones, batch.mask, values, gamma, lambd)

    # Critic phase: masked MSE of BOTH mixed chosen-action critics (paper Eq. 8); only the
    # critics and mixer are stepped. This runs on both streams.
    q1 = mixer(critic1(obs_flat, state_flat, taken_flat).squeeze(-1), state_flat).view(batch_size, horizon)
    q2 = mixer(critic2(obs_flat, state_flat, taken_flat).squeeze(-1), state_flat).view(batch_size, horizon)
    critic_loss = 0.5 * (((q1 - targets) * batch.mask) ** 2).sum() / batch.mask.sum() \
        + 0.5 * (((q2 - targets) * batch.mask) ** 2).sum() / batch.mask.sum()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(critic1.parameters()) + list(critic2.parameters()) + list(mixer.parameters()), GRAD_CLIP,
    )
    critic_opt.step()

    if off:
        return                                                          # off stream stops here

    # Actor phase (paper Eq. 10): re-run the policy under full communication with dropout
    # and push its softmax through critic1 + mixer; the critic and mixer take part in the
    # graph but only the policy parameters are stepped.
    hidden = agent.init_state(batch_size, dev)
    probs_seq = []
    for t in range(horizon):
        mask_t = agent.comm_mask(batch.obs[:, t], last_actions[:, t], selector_on=False, dropout_p=dropout_p)
        logits_t, hidden = agent(batch.obs[:, t], hidden, mask_t, last_actions[:, t])
        probs_seq.append(torch.softmax(logits_t, dim=-1))
    probs = torch.stack(probs_seq, dim=1)                               # (B, T, n, A)
    q_pi = critic1(obs_flat, state_flat, probs.reshape(batch_size * horizon, n, num_actions)).squeeze(-1)
    objective = mixer(q_pi, state_flat).view(batch_size, horizon)
    entropy = -(probs * probs.clamp_min(1e-10).log()).sum(-1).mean(-1)  # (B, T)
    actor_loss = -(objective * batch.mask).sum() / batch.mask.sum()
    actor_loss = actor_loss - entropy_coef * _normalised_entropy(entropy, batch.mask)
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(_actor_params(agent), GRAD_CLIP)
    actor_opt.step()

    # Selector phase (paper Eq. 11): k_i(s)-scaled SMV labels — target policy, online twin
    # critics' min — over the real steps only, regressed by the learner-side selector copy
    # (synced into the acting agent only at target updates).
    real = batch.mask.bool()
    obs_real = batch.obs[:, :-1][real]                                  # (M, n, obs_dim)
    state_real = state[:, :-1][real]
    hidden_real = torch.stack(pre_hiddens[:-1], dim=1)[real]
    last_real = last_actions[:, :-1][real]
    smv = shapley_message_values(
        target_agent, critic1, critic2, obs_real, state_real, hidden_real, last_real, sample_size=smv_sample_size,
    )
    with torch.no_grad():
        labels = mixer.k(state_real).unsqueeze(-1) * smv                # receiver i's row scaled by k_i(s)
    selector_loss = _selector_loss(learner_selector(agent.build_inputs(obs_real, last_real)), labels)
    selector_opt.zero_grad(set_to_none=True)
    selector_loss.backward()
    torch.nn.utils.clip_grad_norm_(learner_selector.parameters(), GRAD_CLIP)
    selector_opt.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SMS on a cooperative task.")
    parser.add_argument("--env", default="simple_spread")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="sms.pt")
    args = parser.parse_args()
    train(env=args.env, episodes=args.episodes, n_agents=args.n_agents, seed=args.seed, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
