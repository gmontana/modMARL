"""Train paper-correct sparse CommFormer with its released bi-level schedule."""

from __future__ import annotations

import argparse

from examples._transformer_training import train_transformer
from marl_envs import make_env
from modmarl.algorithms.commformer import CommFormerAgent


def train(
    *,
    env: str = "navigation",
    n_agents: int = 3,
    horizon: int = 25,
    episodes: int = 300,
    seed: int = 7,
    embedding_dim: int = 64,
    n_heads: int = 1,
    n_blocks: int = 1,
    sparsity: float = 0.4,
    warmup_updates: int = 10,
    bilevel: bool = True,
    post_stable: bool = False,
    post_ratio: float = 0.5,
    share_actor: bool = False,
    self_loop_add: bool = True,
    learning_rate: float = 5e-4,
    edge_learning_rate: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    rollout_episodes: int = 128,
    ppo_epochs: int = 15,
    num_minibatches: int = 1,
    evaluation_episodes: int = 32,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> dict:
    """Run CommFormer on bounded navigation with the primary-paper static graph."""
    environment = make_env(env, n_agents, horizon, seed)

    def learner_factory() -> CommFormerAgent:
        return CommFormerAgent(
            environment.n_agents,
            environment.obs_dim,
            environment.num_actions,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            sparsity=sparsity,
            warmup_updates=warmup_updates,
            bilevel=bilevel,
            post_stable=post_stable,
            post_ratio=post_ratio,
            share_actor=share_actor,
            self_loop_add=self_loop_add,
            learning_rate=learning_rate,
            edge_learning_rate=edge_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_eps,
            entropy_coef=entropy_coef,
        )

    return train_transformer(
        algorithm="commformer",
        learner_factory=learner_factory,
        env=env,
        n_agents=n_agents,
        horizon=horizon,
        episodes=episodes,
        seed=seed,
        rollout_episodes=rollout_episodes,
        ppo_epochs=ppo_epochs,
        num_minibatches=num_minibatches,
        evaluation_episodes=evaluation_episodes,
        device=device,
        checkpoint=checkpoint,
        algorithm_config={
            "embedding_dim": embedding_dim,
            "n_heads": n_heads,
            "n_blocks": n_blocks,
            "sparsity": sparsity,
            "warmup_updates": warmup_updates,
            "bilevel": bilevel,
            "post_stable": post_stable,
            "post_ratio": post_ratio,
            "relation_enhanced": True,
            "share_actor": share_actor,
            "self_loop_add": self_loop_add,
            "learning_rate": learning_rate,
            "edge_learning_rate": edge_learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_epsilon": clip_eps,
            "entropy_coef": entropy_coef,
            "value_loss_coef": 1.0,
            "max_grad_norm": 10.0,
            "huber_delta": 10.0,
            "value_normalization": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CommFormer.")
    parser.add_argument("--env", default="navigation")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="commformer.pt")
    args = parser.parse_args()
    train(
        env=args.env,
        episodes=args.episodes,
        n_agents=args.n_agents,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
