from __future__ import annotations

import pytest
import torch

pytest.importorskip("gymnasium")

from examples.train_ndq import train
from modmarl.algorithms.ndq import NDQAgent
from modmarl.common.replay import EpisodeReplayBuffer


def _agent(**overrides) -> NDQAgent:
    kwargs = dict(
        n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16,
        message_dim=2, mixer_hidden_dim=8,
        include_agent_id=False, include_last_action=False,
    )
    kwargs.update(overrides)
    return NDQAgent(**kwargs)


class _IndexEncoderStub(torch.nn.Module):
    """means() -> mu[b, i, j, k] = 100*i + 10*j + k, so routing is decodable."""

    def __init__(self, n_agents: int, message_dim: int):
        super().__init__()
        self.n_agents, self.message_dim = n_agents, message_dim

    def means(self, obs):
        i = torch.arange(self.n_agents).view(1, -1, 1, 1) * 100.0
        j = torch.arange(self.n_agents).view(1, 1, -1, 1) * 10.0
        k = torch.arange(self.message_dim).view(1, 1, 1, -1) * 1.0
        return (i + j + k).expand(obs.shape[0], -1, -1, -1).clone()


def test_ndq_message_routing_layout(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    agent.message_encoder.means = _IndexEncoderStub(3, 2).means
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.zeros_like(t))

    m_in, _mu = agent.messages(torch.zeros(1, 3, 5))

    # Receiver j gets the concatenation over senders i (in sender order) of chunk m_ij.
    for receiver in range(3):
        for sender in range(3):
            chunk = m_in[0, receiver, sender * 2:(sender + 1) * 2]
            if sender == receiver:
                assert torch.all(chunk == 0.0)          # self-chunk zeroed (paper semantics)
            else:
                expected = torch.tensor([100.0 * sender + 10.0 * receiver, 100.0 * sender + 10.0 * receiver + 1.0])
                assert torch.equal(chunk, expected)


def test_ndq_message_drop_gates_q_input_exactly(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.zeros_like(t))
    obs = torch.randn(2, 3, 5)

    _m_in_all, mu = agent.messages(obs, drop_threshold=None)
    m_in_cut, _ = agent.messages(obs, drop_threshold=float(mu.detach().abs().max()) + 1.0)
    assert torch.all(m_in_cut == 0.0)

    hidden = agent.init_hidden(2, obs.device)
    q_cut, _ = agent.q_step(obs, m_in_cut, hidden)
    q_zero, _ = agent.q_step(obs, torch.zeros_like(m_in_cut), hidden)
    assert torch.allclose(q_cut, q_zero)

    # Per-bit gating: bits with |mu| below the threshold vanish, others survive.
    threshold = float(mu.detach().abs().flatten().median())
    m_in_partial, _ = agent.messages(obs, drop_threshold=threshold)
    kept = (mu.abs() >= threshold).to(mu.dtype)
    no_self = 1.0 - torch.eye(3).view(1, 3, 3, 1)
    expected = (mu * kept * no_self).permute(0, 2, 1, 3).reshape(2, 3, 6)
    assert torch.allclose(m_in_partial, expected)


def test_ndq_expressiveness_gradient_reaches_encoder_not_q_network() -> None:
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    m_in, _ = agent.messages(obs)
    logits = agent.posterior_logits(obs, m_in)
    labels = torch.randint(0, 4, (2, 3))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.message_encoder.mean_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.message_encoder.posterior.parameters())
    assert all(p.grad is None for p in agent.q_network.parameters())
    assert all(p.grad is None for p in agent.target_q_network.parameters())


def test_ndq_succinctness_kl_closed_form() -> None:
    # KL(N(mu, I) || N(0, I)) with unit sigma is exactly mu^2 / 2 per bit.
    mu = torch.randn(4, 3, 3, 2)
    kl = 0.5 * mu.pow(2)
    normal = torch.distributions.Normal(mu, torch.ones_like(mu))
    standard = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    assert torch.allclose(torch.distributions.kl_divergence(normal, standard), kl, atol=1e-6)


def test_ndq_succinctness_excludes_self_message_diagonal() -> None:
    agent = _agent()
    mu = torch.zeros(1, 1, 3, 3, 2)
    mu[:, :, 0, 0] = 1000.0
    assert agent.succinctness_loss(mu, torch.ones(1, 1)) == 0.0

    mu[:, :, 0, 1] = torch.tensor([3.0, 4.0])
    assert agent.succinctness_loss(mu, torch.ones(1, 1)) == pytest.approx(12.5)


def test_ndq_paper_default_uses_beta_one_e_minus_five() -> None:
    agent = _agent()
    assert agent.communication_weight == pytest.approx(0.1)
    assert agent.succinctness_weight == pytest.approx(1e-5)


def test_ndq_defaults_match_released_controller_and_optimizer() -> None:
    agent = NDQAgent(n_agents=3, obs_dim=5, action_dim=4)
    assert agent.include_agent_id
    assert agent.include_last_action
    assert isinstance(agent.optimizer, torch.optim.RMSprop)
    assert agent.message_encoder.mean_head[0].in_features == 12


def test_ndq_update_labels_and_targets_use_target_networks(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    agent.optimizer.param_groups[0]["lr"] = 0.0
    buffer = EpisodeReplayBuffer(capacity=4, horizon=4, n_agents=3, obs_dim=5)
    import numpy as np

    for terminated in (False, True):
        length = 4 if not terminated else 3
        buffer.add_episode(
            obs=np.random.randn(length + 1, 3, 5).astype(np.float32),
            actions=np.random.randint(0, 4, (length, 3)),
            rewards=np.random.randn(length).astype(np.float32),
            dones=np.eye(1, length, length - 1, dtype=np.float32).ravel() * float(terminated),
        )
    batch = buffer.sample(2, torch.device("cpu"))

    agent.gamma = 0.9
    agent.update(batch)
    snapshot = [p.clone() for p in agent.target_q_network.parameters()]
    agent.update(batch)
    for before, after in zip(snapshot, agent.target_q_network.parameters()):
        assert torch.equal(before, after)   # updates never touch the target nets

    # Perturbing the online Q-net must not leak into the target net the labels come from.
    with torch.no_grad():
        for p in agent.q_network.parameters():
            p.add_(1.0)
    for online, target in zip(agent.q_network.parameters(), agent.target_q_network.parameters()):
        assert not torch.equal(online, target)


def test_ndq_target_schedule_is_owned_by_learner() -> None:
    agent = _agent()
    with torch.no_grad():
        next(agent.q_network.parameters()).add_(1.0)
    assert not agent.update_targets_if_due(199, 200)
    assert agent.update_targets_if_due(200, 200)
    assert all(
        torch.equal(online, target)
        for online, target in zip(agent.q_network.parameters(), agent.target_q_network.parameters())
    )


def test_ndq_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "ndq.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=6,
        seed=5,
        hidden_dim=16,
        message_dim=2,
        mixer_hidden_dim=8,
        buffer_episodes=16,
        batch_episodes=2,
        warmup_episodes=2,
        evaluation_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "ndq"
    assert len(summary["final_evaluation"]["returns"]) == 2
    assert len(summary["message_ablated_evaluation"]["returns"]) == 2
    assert summary["communication_rate"] == 1.0
    assert checkpoint.exists()
    state_dict = torch.load(str(checkpoint), weights_only=True)["model"]
    for prefix in ("message_encoder", "q_network", "mixer"):
        assert any(key.startswith(prefix) for key in state_dict)


def test_ndq_optimizer_covers_posterior_but_not_targets() -> None:
    agent = _agent()
    ids = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    assert all(id(p) in ids for p in agent.message_encoder.posterior.parameters())
    for target in (agent.target_message_encoder, agent.target_q_network, agent.target_mixer):
        assert all(id(p) not in ids for p in target.parameters())


def test_ndq_td_gradient_reaches_message_encoder() -> None:
    # The paper's two-gradient property: with the message losses off (c_beta=0), the
    # TD loss alone must still train the encoder through the sampled message in the
    # Q input — an accidental detach on m_in would silently kill communication learning.
    import numpy as np

    torch.manual_seed(0)
    agent = _agent()
    agent.optimizer.param_groups[0]["lr"] = 0.0
    buffer = EpisodeReplayBuffer(capacity=4, horizon=3, n_agents=3, obs_dim=5)
    buffer.add_episode(
        obs=np.random.randn(4, 3, 5).astype(np.float32),
        actions=np.random.randint(0, 4, (3, 3)),
        rewards=np.random.randn(3).astype(np.float32),
        dones=np.zeros(3, dtype=np.float32),
    )
    agent.gamma = 0.9
    agent.communication_weight = 0.0
    agent.succinctness_weight = 0.0
    agent.update(buffer.sample(1, torch.device("cpu")))

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.message_encoder.mean_head.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in agent.message_encoder.posterior.parameters())


def test_ndq_aux_loss_normalisation_golden(monkeypatch) -> None:
    # Freeze the scale convention: expressiveness = CE summed over agents / mask.sum();
    # succinctness = per-step sum of mu^2/2 masked / mask.sum().
    import numpy as np

    torch.manual_seed(0)
    agent = _agent(communication_weight=1.0, succinctness_weight=1.0)
    agent.optimizer.param_groups[0]["lr"] = 0.0
    buffer = EpisodeReplayBuffer(capacity=4, horizon=2, n_agents=3, obs_dim=5)
    buffer.add_episode(
        obs=np.random.randn(3, 3, 5).astype(np.float32),
        actions=np.random.randint(0, 4, (2, 3)),
        rewards=np.zeros(2, dtype=np.float32),
        dones=np.zeros(2, dtype=np.float32),
    )
    batch = buffer.sample(1, torch.device("cpu"))
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.zeros_like(t))

    losses = {}
    real_backward = torch.Tensor.backward

    def spy(self, *args, **kwargs):
        losses["total"] = float(self.detach())
        return real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    agent.gamma = 0.9
    agent.update(batch)

    # Recompute the pieces by hand with the same zeroed noise.
    with torch.no_grad():
        hidden = agent.init_hidden(1, torch.device("cpu"))
        target_hidden = agent.init_hidden(1, torch.device("cpu"))
        qs, tqs, mus, m_ins = [], [], [], []
        for t in range(3):
            m_in, mu = agent.messages(batch.obs[:, t])
            q, hidden = agent.q_step(batch.obs[:, t], m_in, hidden)
            tm, _ = agent.messages(batch.obs[:, t], target=True)
            tq, target_hidden = agent.q_step(batch.obs[:, t], tm, target_hidden, target=True)
            qs.append(q)
            tqs.append(tq)
            mus.append(mu)
            m_ins.append(m_in)
        mask_sum = float(batch.mask.sum())
        ce = 0.0
        for t in range(2):
            logits = agent.posterior_logits(batch.obs[:, t], m_ins[t])
            labels = tqs[t].argmax(-1)
            ce += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, 4), labels.reshape(-1), reduction="sum",
            ) * batch.mask[0, t])
        expressiveness = ce / mask_sum
        no_self = 1.0 - torch.eye(3).view(1, 3, 3, 1)
        succinctness = sum(
            float(0.5 * (mus[t].pow(2) * no_self).sum() * batch.mask[0, t]) for t in range(2)
        ) / mask_sum
        chosen = torch.stack(qs[:-1], 1).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        state = batch.obs.reshape(1, 3, -1)
        q_tot = agent.mixer(chosen.reshape(-1, 3), state[:, :-1].reshape(2, -1)).view(1, 2)
        nxt = torch.stack(tqs[1:], 1).gather(-1, torch.stack(qs[1:], 1).argmax(-1, keepdim=True)).squeeze(-1)
        nxt_tot = agent.target_mixer(nxt.reshape(-1, 3), state[:, 1:].reshape(2, -1)).view(1, 2)
        y = batch.rewards + 0.9 * (1.0 - batch.dones) * nxt_tot
        td = float((((q_tot - y) * batch.mask) ** 2).sum() / mask_sum)

    expected = td + 1.0 * (expressiveness + 1.0 * succinctness)
    assert losses["total"] == pytest.approx(expected, rel=1e-4)
