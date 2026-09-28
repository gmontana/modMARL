from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("gymnasium")

from examples import train_masia
from examples.train_masia import train
from modmarl.algorithms.masia import InformationAggregationEncoder, MASIAAgent
from modmarl.common.replay import EpisodeReplayBuffer


def _agent(**overrides) -> MASIAAgent:
    kwargs = dict(
        n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16, enc_hidden_dim=8, attn_dim=8,
        z_slot_dim=4, ob_embed_dim=8, spr_dim=8, model_hidden_dim=16, mixer_hidden_dim=8,
    )
    kwargs.update(overrides)
    return MASIAAgent(**kwargs)


def _episode_batch(agent, horizon: int, length: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    buffer = EpisodeReplayBuffer(capacity=4, horizon=horizon, n_agents=3, obs_dim=5)
    buffer.add_episode(
        obs=rng.standard_normal((length + 1, 3, 5)).astype(np.float32),
        actions=rng.integers(0, 4, (length, 3)),
        rewards=rng.standard_normal(length).astype(np.float32),
        dones=np.zeros(length, dtype=np.float32),
    )
    return buffer.sample(1, torch.device("cpu"))


def test_masia_released_inputs_and_mixer_selection() -> None:
    obs = torch.randn(2, 3, 5)
    previous = F.one_hot(torch.randint(0, 4, (2, 3)), 4).float()
    agent = _agent(include_previous_action=True)
    inputs = agent.build_inputs(obs, previous)
    assert torch.equal(inputs[..., :5], obs)
    assert torch.equal(inputs[..., 5:9], previous)
    assert torch.equal(inputs[0, :, 9:], torch.eye(3))
    assert isinstance(agent.mixer.hyper_w1, torch.nn.Sequential)
    vdn = _agent(mixer="vdn")
    assert torch.equal(
        vdn.mixer(torch.tensor([[1.0, 2.0, 3.0]]), torch.empty(1, 15)),
        torch.tensor([6.0]),
    )


def test_masia_aggregation_is_permutation_equivariant() -> None:
    # The paper's "permutation invariant" encoder as realised by integration design (c):
    # each z slot is a set function of the received messages, so permuting the agents
    # permutes the slots. The Q path is agent-symmetric given the shared z (no agent-ID
    # inputs — deviation 1). The composed forward is deliberately NOT equivariant: the
    # gate and fc1 weights index absolute slot positions, matching the official net.
    torch.manual_seed(0)
    agent = _agent(include_agent_id=False)
    obs = torch.randn(2, 3, 5)
    q_hidden = torch.randn(2, 3, 16)
    enc_hidden = torch.randn(2, 3, 8)
    perm = torch.tensor([2, 0, 1])

    z, new_enc = agent.enc_forward(obs, enc_hidden)
    z_p, new_enc_p = agent.enc_forward(obs[:, perm], enc_hidden[:, perm])
    assert torch.allclose(z_p.view(2, 3, 4), z.view(2, 3, 4)[:, perm], atol=1e-6)
    assert torch.allclose(new_enc_p, new_enc[:, perm], atol=1e-6)

    q, new_q_hidden = agent.q_forward(obs, z, q_hidden)
    q_p, new_q_hidden_p = agent.q_forward(obs[:, perm], z, q_hidden[:, perm])
    assert torch.allclose(q_p, q[:, perm], atol=1e-5)
    assert torch.allclose(new_q_hidden_p, new_q_hidden[:, perm], atol=1e-5)


def test_masia_repr_loss_reaches_encoder_but_not_q_head() -> None:
    # L_ae + L_m + L_r on a tiny unpadded episode: gradients reach the encoder, decoder,
    # SPR heads and transition model, but never the gate/Q trunk or the mixer (the
    # official stop-gradient direction; the reverse direction is pinned separately).
    torch.manual_seed(0)
    agent = _agent()
    steps = 4
    obs = torch.randn(2, steps, 3, 5)
    actions_onehot = F.one_hot(torch.randint(0, 4, (2, steps - 1, 3)), 4).float()
    rewards = torch.randn(2, steps - 1)

    _, enc_hidden = agent.init_state(2, torch.device("cpu"))
    _, target_enc_hidden = agent.init_state(2, torch.device("cpu"))
    zs, target_projs = [], []
    for t in range(steps):
        z, enc_hidden = agent.enc_forward(obs[:, t], enc_hidden)
        proj, _, target_enc_hidden = agent.target_project_enc(obs[:, t], target_enc_hidden)
        zs.append(z)
        target_projs.append(proj)
    z_seq = torch.stack(zs, dim=1)                              # (B, steps, n*z_slot)
    target_proj = torch.stack(target_projs, dim=1)

    l_ae = ((agent.encoder.decode(z_seq) - obs.reshape(2, steps, -1)) ** 2).mean()
    l_m, l_r = 0.0, 0.0
    rollout = z_seq
    for k in range(3):                                          # K = 2 plus the k=0 term
        if k > 0:
            rollout = agent.transition_model(rollout[:, :-1], actions_onehot[:, k - 1:])
        l_m = l_m + ((agent.project(rollout) - target_proj[:, k:]) ** 2).sum(dim=-1).mean()
        predicted_r = agent.transition_model.predict_reward(rollout[:, :-1]).squeeze(-1)
        l_r = l_r + ((predicted_r - rewards[:, k:]) ** 2).mean()
    (l_ae + l_m + l_r).backward()

    for module in (agent.encoder.query, agent.encoder.key, agent.encoder.value,
                   agent.encoder.gru, agent.encoder.bottleneck, agent.encoder.decoder,
                   agent.projection, agent.predictor, agent.transition_model):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for module in (agent.q_network.gate, agent.q_network.ob_fc, agent.q_network.fc1,
                   agent.q_network.gru, agent.q_network.head, agent.mixer):
        assert all(p.grad is None for p in module.parameters())
    assert all(p.grad is None for p in agent.target_encoder.parameters())


def test_masia_td_loss_reaches_encoder() -> None:
    # rl_signal semantics: with the representation losses off, the TD loss alone still
    # trains the encoder through z in the Q input (an accidental detach would silently
    # reduce MASIA to recurrent QMIX) — the reverse of the repr-loss separation test.
    torch.manual_seed(0)
    agent = _agent()
    opt = torch.optim.Adam(agent.q_network.parameters(), lr=0.0)
    batch = _episode_batch(agent, horizon=3, length=3)

    train_masia._update(agent, opt, batch, gamma=0.9, repr_coef=0.0, spr_coef=1.0,
                        rew_pred_coef=1.0, pred_len=2)

    for module in (agent.encoder.query, agent.encoder.key, agent.encoder.value, agent.q_network.gate):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for module in (agent.encoder.decoder, agent.predictor):
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in module.parameters())


def test_masia_spr_targets_are_gradient_free_and_hard_copied() -> None:
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    _, enc_hidden = agent.init_state(2, torch.device("cpu"))

    proj, z, _ = agent.target_project_enc(obs, enc_hidden)
    assert not proj.requires_grad and not z.requires_grad

    # Perturbing the online encoder/projection must not leak into the targets ...
    with torch.no_grad():
        for p in list(agent.encoder.parameters()) + list(agent.projection.parameters()):
            p.add_(1.0)
    proj_after, _, _ = agent.target_project_enc(obs, enc_hidden)
    assert torch.equal(proj, proj_after)

    # ... until update_targets(), which hard-copies (momentum_tau = 1): targets then
    # match the online encoder + projection (not the predictor) exactly.
    agent.update_targets()
    proj_copied, _, _ = agent.target_project_enc(obs, enc_hidden)
    with torch.no_grad():
        z_online, _ = agent.enc_forward(obs, enc_hidden)
        expected = agent.projection(z_online)
    assert torch.equal(proj_copied, expected)
    assert not torch.equal(proj_copied, proj)


def test_masia_focus_gate_filters_dimensions() -> None:
    # Paper §3.1: near-zero gate weights filter those z dimensions out entirely.
    torch.manual_seed(0)
    agent = _agent()
    with torch.no_grad():
        agent.q_network.gate.weight.zero_()
        agent.q_network.gate.bias[:6] = -1e4    # sigmoid == 0: blocked dims
        agent.q_network.gate.bias[6:] = 1e4     # sigmoid == 1: open dims
    obs = torch.randn(2, 3, 5)
    q_hidden, _ = agent.init_state(2, torch.device("cpu"))
    z = torch.randn(2, 12)

    q, _ = agent.q_forward(obs, z, q_hidden)
    z_blocked = z.clone()
    z_blocked[:, :6] += 100.0
    q_blocked, _ = agent.q_forward(obs, z_blocked, q_hidden)
    assert torch.equal(q_blocked, q)            # blocked dims cannot move Q at all

    z_open = z.clone()
    z_open[:, 6:] += 1.0
    q_open, _ = agent.q_forward(obs, z_open, q_hidden)
    assert not torch.allclose(q_open, q)


def test_masia_reconstruction_target_is_global_state(monkeypatch) -> None:
    # decode(encode(obs)) covers the full concat-obs state, and the trainer's L_ae is
    # the masked per-step MSE over the real observation steps (padded steps contribute
    # zero). Extracted from _update by differencing two runs on a frozen optimizer.
    torch.manual_seed(0)
    agent = _agent()
    _, enc_hidden = agent.init_state(2, torch.device("cpu"))
    z, _ = agent.enc_forward(torch.randn(2, 3, 5), enc_hidden)
    assert tuple(agent.encoder.decode(z).shape) == (2, 15)

    opt = torch.optim.Adam(agent.q_network.parameters(), lr=0.0)
    batch = _episode_batch(agent, horizon=5, length=3)          # two padded steps

    totals: list[float] = []
    real_backward = torch.Tensor.backward

    def spy(self, *args, **kwargs):
        totals.append(float(self.detach()))
        return real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    train_masia._update(agent, opt, batch, gamma=0.9, repr_coef=1.0, spr_coef=0.0,
                        rew_pred_coef=0.0, pred_len=2)
    train_masia._update(agent, opt, batch, gamma=0.9, repr_coef=0.0, spr_coef=0.0,
                        rew_pred_coef=0.0, pred_len=2)
    recon = totals[0] - totals[1]                               # td cancels: lr=0, no RNG

    with torch.no_grad():
        _, enc_hidden = agent.init_state(1, torch.device("cpu"))
        state = batch.obs.reshape(1, 6, -1)
        obs_mask = [1.0] + [float(m) for m in batch.mask[0]]    # final real obs included
        numerator = 0.0
        for t in range(6):
            z_t, enc_hidden = agent.enc_forward(batch.obs[:, t], enc_hidden)
            numerator += float(((agent.encoder.decode(z_t) - state[:, t]) ** 2).mean()) * obs_mask[t]
    assert recon == pytest.approx(numerator / sum(obs_mask), rel=1e-4)


def test_masia_latent_rollout_alignment(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()

    # Shapes: each rollout level drops one step; residual path pinned by zeroing the
    # last delta layer, which must make the model the exact identity in z.
    z = torch.randn(2, 5, 12)
    actions_onehot = F.one_hot(torch.randint(0, 4, (2, 4, 3)), 4).float()
    level1 = agent.transition_model(z[:, :-1], actions_onehot)
    level2 = agent.transition_model(level1[:, :-1], actions_onehot[:, 1:])
    assert tuple(level1.shape) == (2, 4, 12)
    assert tuple(level2.shape) == (2, 3, 12)
    assert torch.isfinite(level2).all()
    assert tuple(agent.transition_model.predict_reward(z).shape) == (2, 5, 1)
    with torch.no_grad():
        agent.transition_model.delta_mlp[-1].weight.zero_()
        agent.transition_model.delta_mlp[-1].bias.zero_()
    assert torch.equal(agent.transition_model(z, F.one_hot(torch.randint(0, 4, (2, 5, 3)), 4).float()), z)

    # Golden alignment against _update: the k-th term pairs the prediction rolled out
    # from base step t with step t+k, and shifts the mask identically.
    agent = _agent()
    opt = torch.optim.Adam(agent.q_network.parameters(), lr=0.0)
    batch = _episode_batch(agent, horizon=4, length=3)          # one padded step

    totals: list[float] = []
    real_backward = torch.Tensor.backward

    def spy(self, *args, **kwargs):
        totals.append(float(self.detach()))
        return real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    for spr_coef, rew_pred_coef in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        train_masia._update(agent, opt, batch, gamma=0.9, repr_coef=1.0,
                            spr_coef=spr_coef, rew_pred_coef=rew_pred_coef, pred_len=2)
    spr_value = totals[1] - totals[0]
    reward_value = totals[2] - totals[0]

    with torch.no_grad():
        _, enc_hidden = agent.init_state(1, torch.device("cpu"))
        _, target_enc_hidden = agent.init_state(1, torch.device("cpu"))
        zs, target_projs = [], []
        for t in range(5):
            z_t, enc_hidden = agent.enc_forward(batch.obs[:, t], enc_hidden)
            proj_t, _, target_enc_hidden = agent.target_project_enc(batch.obs[:, t], target_enc_hidden)
            zs.append(z_t)
            target_projs.append(proj_t)
        actions_onehot = F.one_hot(batch.actions, 4).float()
        obs_mask = [1.0] + [float(m) for m in batch.mask[0]]
        levels = [zs]                                           # levels[k][i] predicts step k+i
        for k in (1, 2):
            levels.append([
                agent.transition_model(levels[k - 1][i], actions_onehot[:, k - 1 + i])
                for i in range(len(levels[k - 1]) - 1)
            ])
        spr_expected, reward_expected = 0.0, 0.0
        for k in range(3):
            spr_expected += sum(
                float(((agent.project(levels[k][i]) - target_projs[k + i]) ** 2).sum()) * obs_mask[k + i]
                for i in range(len(levels[k]))
            ) / sum(obs_mask[k:])
            reward_expected += sum(
                float(agent.transition_model.predict_reward(levels[k][i]).squeeze()
                      - batch.rewards[0, k + i]) ** 2 * float(batch.mask[0, k + i])
                for i in range(len(levels[k]) - 1)
            ) / float(batch.mask[0, k:].sum())
    assert spr_value == pytest.approx(spr_expected, rel=1e-4)
    assert reward_value == pytest.approx(reward_expected, rel=1e-4)


def test_masia_two_hidden_streams_thread() -> None:
    # The scaffold invariant extended to the second stream: a greedy rollout threading
    # (q_hidden, enc_hidden) is reproduced exactly by replaying from zeros with the
    # decomposed encode-then-Q path _update uses.
    torch.manual_seed(0)
    agent = _agent()
    obs_seq = torch.randn(1, 5, 3, 5)

    with torch.no_grad():
        q_hidden, enc_hidden = agent.init_state(1, torch.device("cpu"))
        rollout_qs = []
        for t in range(5):
            q, q_hidden, enc_hidden = agent(obs_seq[:, t], q_hidden, enc_hidden)
            rollout_qs.append(q)
        assert not torch.equal(q_hidden, torch.zeros_like(q_hidden))
        assert not torch.equal(enc_hidden, torch.zeros_like(enc_hidden))

        q_hidden, enc_hidden = agent.init_state(1, torch.device("cpu"))
        for t in range(5):
            z, enc_hidden = agent.enc_forward(obs_seq[:, t], enc_hidden)
            q, q_hidden = agent.q_forward(obs_seq[:, t], z, q_hidden)
            assert torch.equal(q, rollout_qs[t])


def test_masia_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "masia.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=6,
        seed=5,
        hidden_dim=16,
        enc_hidden_dim=8,
        z_slot_dim=4,
        spr_dim=8,
        mixer_hidden_dim=8,
        buffer_episodes=16,
        batch_episodes=2,
        warmup_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "masia"
    assert summary["communication_rate"] == 1.0
    assert checkpoint.exists()
    state_dict = torch.load(str(checkpoint), weights_only=True)
    for prefix in ("encoder", "q_network", "projection", "predictor", "transition_model", "mixer"):
        assert any(key.startswith(prefix) for key in state_dict)


def _terminated_batch():
    import numpy as np

    rng = np.random.default_rng(3)
    buffer = EpisodeReplayBuffer(capacity=4, horizon=4, n_agents=3, obs_dim=5)
    buffer.add_episode(
        obs=rng.standard_normal((3, 3, 5)).astype(np.float32),
        actions=rng.integers(0, 4, (2, 3)),
        rewards=np.array([0.5, 2.0], dtype=np.float32),
        dones=np.array([0.0, 1.0], dtype=np.float32),
    )
    return buffer.sample(1, torch.device("cpu"))


def _recompute_td(agent, batch, gamma, *, double_q: bool):
    with torch.no_grad():
        batch_size, horizon = batch.actions.shape[:2]
        q_hidden, enc_hidden = agent.init_state(batch_size, torch.device("cpu"))
        tq_hidden, tenc_hidden = agent.init_state(batch_size, torch.device("cpu"))
        online_q, target_q = [], []
        for t in range(horizon + 1):
            obs_t = batch.obs[:, t]
            z, enc_hidden = agent.enc_forward(obs_t, enc_hidden)
            q, q_hidden = agent.q_forward(obs_t, z, q_hidden)
            _, tz, tenc_hidden = agent.target_project_enc(obs_t, tenc_hidden)
            qt, tq_hidden = agent.q_forward(obs_t, tz, tq_hidden, target=True)
            online_q.append(q)
            target_q.append(qt)
        online_q = torch.stack(online_q, dim=1)
        target_q = torch.stack(target_q, dim=1)
        chosen = online_q[:, :-1].gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        state = batch.obs.reshape(batch_size, horizon + 1, -1)
        q_tot = agent.mixer(chosen.reshape(-1, 3), state[:, :-1].reshape(-1, state.shape[-1])).view(batch_size, horizon)
        if double_q:
            nxt_idx = online_q[:, 1:].argmax(dim=-1, keepdim=True)
        else:
            nxt_idx = target_q[:, 1:].argmax(dim=-1, keepdim=True)
        nxt = target_q[:, 1:].gather(-1, nxt_idx).squeeze(-1)
        nxt_tot = agent.target_mixer(nxt.reshape(-1, 3), state[:, 1:].reshape(-1, state.shape[-1])).view(batch_size, horizon)
        y = batch.rewards + gamma * (1.0 - batch.dones) * nxt_tot
        return ((((q_tot - y) * batch.mask) ** 2).sum() / batch.mask.sum()), y


def test_masia_td_target_respects_termination(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    opt = torch.optim.Adam(train_masia._online_params(agent), lr=0.0)
    batch = _terminated_batch()

    losses = {}
    real_backward = torch.Tensor.backward

    def spy(self, *args, **kwargs):
        losses["total"] = float(self.detach())
        return real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    train_masia._update(agent, opt, batch, gamma=0.9, repr_coef=0.0, spr_coef=1.0, rew_pred_coef=1.0, pred_len=2)

    expected, y = _recompute_td(agent, batch, 0.9, double_q=True)
    assert losses["total"] == pytest.approx(float(expected), rel=1e-5)
    # The terminated step bootstraps nothing: y there is exactly its reward.
    assert float(y[0, 1]) == pytest.approx(2.0)


def test_masia_double_q_uses_online_argmax_with_target_values(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    # Decouple online and target argmaxes so double-Q and plain target-max differ.
    with torch.no_grad():
        for p in agent.q_network.head.parameters():
            p.add_(torch.randn_like(p))
    opt = torch.optim.Adam(train_masia._online_params(agent), lr=0.0)
    batch = _episode_batch(agent, horizon=4, length=4)

    losses = {}
    real_backward = torch.Tensor.backward

    def spy(self, *args, **kwargs):
        losses["total"] = float(self.detach())
        return real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    train_masia._update(agent, opt, batch, gamma=0.9, repr_coef=0.0, spr_coef=1.0, rew_pred_coef=1.0, pred_len=2)

    double_q_loss, _ = _recompute_td(agent, batch, 0.9, double_q=True)
    plain_max_loss, _ = _recompute_td(agent, batch, 0.9, double_q=False)
    assert losses["total"] == pytest.approx(float(double_q_loss), rel=1e-5)
    assert float(double_q_loss) != pytest.approx(float(plain_max_loss), rel=1e-5)


def test_masia_update_targets_syncs_q_network_and_mixer() -> None:
    torch.manual_seed(0)
    agent = _agent()
    with torch.no_grad():
        for p in train_masia._online_params(agent):
            p.add_(1.0)

    for online, target in zip(agent.q_network.parameters(), agent.target_q_network.parameters()):
        assert not torch.equal(online, target)
    agent.update_targets()
    for module, target in (
        (agent.q_network, agent.target_q_network),
        (agent.mixer, agent.target_mixer),
        (agent.encoder, agent.target_encoder),
        (agent.projection, agent.target_projection),
    ):
        for online, copied in zip(module.parameters(), target.parameters()):
            assert torch.equal(online, copied)


def test_masia_full_forward_is_not_permutation_equivariant() -> None:
    # Deliberate property matching the official net: the focusing gate and fc1 index
    # absolute z slots, so the COMPOSED forward is identity-aware — "fixing" the gate
    # to reindex slots would silently change the algorithm.
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    q_hidden, enc_hidden = agent.init_state(2, obs.device)
    perm = torch.tensor([2, 0, 1])

    q, _, _ = agent(obs, q_hidden, enc_hidden)
    q_perm, _, _ = agent(obs[:, perm], q_hidden, enc_hidden)
    assert not torch.allclose(q_perm, q[:, perm], atol=1e-5)


def test_aggregation_encoder_permutes_z_slots_with_the_agents() -> None:
    # The docstring's structural claim: slot i of z is a set function of the other
    # agents' messages, so permuting the agent axis must permute the z_slot_dim-sized
    # slots of z rather than change their contents. This holds only if the attention is
    # equivariant, the GRU integration cell is shared, and the bottleneck is shared —
    # a per-agent bottleneck or a positional embedding would break it.
    torch.manual_seed(53)
    n_agents, obs_dim, z_slot, enc_hidden_dim = 4, 3, 5, 6
    encoder = InformationAggregationEncoder(
        n_agents, obs_dim, attn_dim=8, enc_hidden_dim=enc_hidden_dim, z_slot_dim=z_slot,
    ).double()

    obs = torch.randn(2, n_agents, obs_dim, dtype=torch.float64)
    hidden = torch.randn(2, n_agents, enc_hidden_dim, dtype=torch.float64)
    perm = torch.tensor([2, 0, 3, 1])

    z, _ = encoder.encode(obs, hidden)
    z_perm, _ = encoder.encode(obs[:, perm], hidden[:, perm])

    slots = z.view(2, n_agents, z_slot)
    slots_perm = z_perm.view(2, n_agents, z_slot)
    torch.testing.assert_close(slots_perm, slots[:, perm], rtol=0.0, atol=1e-12)

    # Non-vacuity: the slots must actually differ from one another.
    assert not torch.allclose(slots[:, 0], slots[:, 1])
