from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from modmarl.algorithms.cdc import CDCPolicy, ReleasedCDCCritic
from modmarl.algorithms.commnet import CommNetAgent
from modmarl.algorithms.ddpg import DDPGAgent, DDPGConfig
from modmarl.algorithms.maac import AttentionCritic, MAACAgent
from modmarl.algorithms.maddpg import MADDPGAgent
from modmarl.algorithms.marc import MARCAgent, MARCRelationalCritic


def _assert_loss_decreases(loss_fn, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    before = float(loss_fn().detach().item())
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    after = float(loss_fn().detach().item())
    assert after < before
    return before, after


def _freeze_module(module: torch.nn.Module) -> list[bool]:
    previous = []
    for parameter in module.parameters():
        previous.append(parameter.requires_grad)
        parameter.requires_grad_(False)
    return previous


def _restore_module(module: torch.nn.Module, previous: list[bool]) -> None:
    for parameter, enabled in zip(module.parameters(), previous):
        parameter.requires_grad_(enabled)


def test_ddpg_single_update_reduces_critic_and_actor_losses() -> None:
    torch.manual_seed(0)
    agent = DDPGAgent(
        obs_dim=5,
        action_dim=4,
        hidden_dims=(32, 24),
        config=DDPGConfig(critic_weight_decay=0.0),
    )
    batch_size = 32
    obs = torch.randn(batch_size, 5)
    next_obs = torch.randn(batch_size, 5)
    actions = torch.rand(batch_size, 4)
    rewards = torch.randn(batch_size)
    dones = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=2e-2)

    def critic_loss_fn():
        with torch.no_grad():
            target_next_actions = agent.target_actor(next_obs)
            targets = rewards + 0.99 * (1.0 - dones) * agent.target_critic(next_obs, target_next_actions)
        return F.mse_loss(agent.critic(obs, actions), targets)

    _assert_loss_decreases(critic_loss_fn, critic_optimizer)

    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=1e-2)
    previous = _freeze_module(agent.critic)

    def actor_loss_fn():
        return -agent.critic(obs, agent.actor(obs)).mean()

    _assert_loss_decreases(actor_loss_fn, actor_optimizer)
    _restore_module(agent.critic, previous)


def test_commnet_single_update_changes_the_on_policy_network() -> None:
    torch.manual_seed(0)
    agent = CommNetAgent(obs_dim=5, action_dim=4, hidden_dim=32)
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    update = agent.update(
        torch.randn(8, 6, 3, 5),
        torch.randint(0, 4, (8, 6, 3)),
        torch.randn(8, 6),
        torch.ones(8, 6, 3),
    )
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.actor.parameters()))
    assert torch.isfinite(torch.tensor(update.policy_loss))
    assert torch.isfinite(torch.tensor(update.baseline_loss))


def test_cdc_single_update_reduces_critic_and_actor_losses() -> None:
    torch.manual_seed(0)
    actor = CDCPolicy(
        obs_dim=5,
        action_dim=4,
        message_dim=16,
    )
    critic = ReleasedCDCCritic(obs_dim=5, action_dim=4, hidden_dim=32)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    batch_size = 24
    n_agents = 3
    obs = torch.randn(batch_size, n_agents, 5)
    next_obs = torch.randn(batch_size, n_agents, 5)
    action_idx = torch.randint(0, 4, (batch_size, n_agents))
    actions = F.one_hot(action_idx, num_classes=4).to(dtype=torch.float32)
    rewards = torch.randn(batch_size)
    dones = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2e-2)

    def critic_loss_fn():
        with torch.no_grad():
            next_actions, _, _ = target_actor.sample_gumbel(next_obs, deterministic=True)
            targets = rewards + 0.95 * (1.0 - dones) * target_critic(next_obs, next_actions)
        return F.mse_loss(critic(obs, actions), targets)

    _assert_loss_decreases(critic_loss_fn, critic_optimizer)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
    previous = _freeze_module(critic)

    def actor_loss_fn():
        torch.manual_seed(0)
        actor_actions, _, _ = actor.sample_gumbel(obs, temperature=0.7, hard=False, deterministic=False)
        return -critic(obs, actor_actions).mean()

    _assert_loss_decreases(actor_loss_fn, actor_optimizer)
    _restore_module(critic, previous)


def test_maddpg_single_update_reduces_critic_and_actor_losses() -> None:
    torch.manual_seed(0)
    n_agents = 3
    agents = [MADDPGAgent(n_agents=n_agents, obs_dim=5, action_dim=4, hidden_dim=32) for _ in range(n_agents)]
    batch_size = 24
    obs = torch.randn(batch_size, n_agents, 5)
    next_obs = torch.randn(batch_size, n_agents, 5)
    action_idx = torch.randint(0, 4, (batch_size, n_agents))
    actions = F.one_hot(action_idx, num_classes=4).to(dtype=torch.float32)
    rewards = torch.randn(batch_size)
    dones = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    critic_optimizer = torch.optim.Adam(agents[0].critic.parameters(), lr=2e-2)

    def critic_loss_fn():
        with torch.no_grad():
            target_next_actions = torch.stack(
                [agent.target_actor.sample(next_obs[:, agent_id], deterministic=True)[0] for agent_id, agent in enumerate(agents)],
                dim=1,
            )
            targets = rewards + 0.95 * (1.0 - dones) * agents[0].target_critic(next_obs, target_next_actions)
        return F.mse_loss(agents[0].critic(obs, actions), targets)

    _assert_loss_decreases(critic_loss_fn, critic_optimizer)

    actor_optimizer = torch.optim.Adam(agents[0].actor.parameters(), lr=1e-2)
    previous = _freeze_module(agents[0].critic)

    def actor_loss_fn():
        torch.manual_seed(0)
        current_actions = []
        for agent_id, agent in enumerate(agents):
            if agent_id == 0:
                one_hot, _, _ = agent.actor.sample(
                    obs[:, agent_id],
                    temperature=0.7,
                    hard=False,
                    deterministic=False,
                )
            else:
                one_hot, _, _ = agent.actor.sample(obs[:, agent_id], deterministic=True)
                one_hot = one_hot.detach()
            current_actions.append(one_hot)
        current_actions_tensor = torch.stack(current_actions, dim=1)
        return -agents[0].critic(obs, current_actions_tensor).mean()

    _assert_loss_decreases(actor_loss_fn, actor_optimizer)
    _restore_module(agents[0].critic, previous)


def test_maac_single_update_reduces_critic_loss() -> None:
    torch.manual_seed(0)
    n_agents = 3
    agents = [MAACAgent(obs_dim=5, action_dim=4, hidden_dim=32) for _ in range(n_agents)]
    critic = AttentionCritic(n_agents=n_agents, obs_dim=5, action_dim=4, hidden_dim=32, attend_heads=4)
    target_critic = copy.deepcopy(critic)

    batch_size = 24
    obs = torch.randn(batch_size, n_agents, 5)
    next_obs = torch.randn(batch_size, n_agents, 5)
    action_idx = torch.randint(0, 4, (batch_size, n_agents))
    actions = F.one_hot(action_idx, num_classes=4).to(dtype=torch.float32)
    rewards = torch.randn(batch_size).unsqueeze(-1).expand(-1, n_agents)
    dones = torch.randint(0, 2, (batch_size, n_agents), dtype=torch.float32)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)

    def critic_loss_fn():
        with torch.no_grad():
            torch.manual_seed(0)
            next_actions = []
            next_log_pis = []
            for agent_id, agent in enumerate(agents):
                action_one_hot, _, _, _, _, chosen_log_prob, _ = agent.target_actor.sample(
                    next_obs[:, agent_id],
                )
                next_actions.append(action_one_hot)
                next_log_pis.append(chosen_log_prob)
            next_actions_tensor = torch.stack(next_actions, dim=1)
            next_log_pis_tensor = torch.stack(next_log_pis, dim=1)
            target_q_taken = target_critic(next_obs, next_actions_tensor).q_taken
            target_values = target_q_taken - 0.01 * next_log_pis_tensor
            target_q = rewards + 0.99 * (1.0 - dones) * target_values

        critic_q_taken = critic(obs, actions).q_taken
        return F.mse_loss(critic_q_taken, target_q)

    _assert_loss_decreases(critic_loss_fn, critic_optimizer)


def test_marc_single_update_reduces_critic_loss() -> None:
    torch.manual_seed(0)
    n_agents = 2
    agents = [MARCAgent(obs_dim=12, action_dim=6, hidden_dim=32) for _ in range(n_agents)]
    critic = MARCRelationalCritic(
        n_agents=n_agents,
        node_feature_dim=6,
        action_dim=6,
        hidden_dim=32,
        embed_dim=32,
        num_relations=6,
        num_relational_layers=1,
    )
    target_critic = copy.deepcopy(critic)

    batch_size = 20
    next_obs = torch.randn(batch_size, n_agents, 12)
    node_features = torch.randn(batch_size, n_agents, 4, 6)
    next_node_features = torch.randn(batch_size, n_agents, 4, 6)
    relations = torch.randint(0, 2, (batch_size, 6, 4, 4), dtype=torch.float32)
    next_relations = torch.randint(0, 2, (batch_size, 6, 4, 4), dtype=torch.float32)
    action_idx = torch.randint(0, 6, (batch_size, n_agents))
    actions = F.one_hot(action_idx, num_classes=6).to(dtype=torch.float32)
    rewards = torch.randn(batch_size).unsqueeze(-1).expand(-1, n_agents)
    dones = torch.randint(0, 2, (batch_size, n_agents), dtype=torch.float32)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)

    def critic_loss_fn():
        with torch.no_grad():
            torch.manual_seed(0)
            next_actions = []
            next_log_pis = []
            for agent_id, agent in enumerate(agents):
                action_one_hot, _, _, _, _, chosen_log_prob, _ = agent.target_actor.sample(
                    next_obs[:, agent_id],
                    deterministic=False,
                )
                next_actions.append(action_one_hot)
                next_log_pis.append(chosen_log_prob)
            next_actions_tensor = torch.stack(next_actions, dim=1)
            next_log_pis_tensor = torch.stack(next_log_pis, dim=1)
            target_q_taken, _ = target_critic(
                next_node_features,
                next_relations,
                next_actions_tensor,
                return_all_q=False,
            )
            target_values = target_q_taken - 0.01 * next_log_pis_tensor
            target_q = rewards + 0.99 * (1.0 - dones) * target_values

        critic_q_taken, _ = critic(node_features, relations, actions, return_all_q=False)
        return F.mse_loss(critic_q_taken, target_q)

    _assert_loss_decreases(critic_loss_fn, critic_optimizer)
