from __future__ import annotations

import inspect
import math

import pytest
import torch

pytest.importorskip("gymnasium")

from examples import train_maic
from examples.train_maic import train
from modmarl.algorithms.maic import MAICAgent


def _agent(**overrides) -> MAICAgent:
    kwargs = dict(n_agents=3, obs_dim=5, action_dim=4, hidden_dim=16, latent_dim=4, attention_dim=8, mixer_hidden_dim=8)
    kwargs.update(overrides)
    return MAICAgent(**kwargs)


def _step(agent, obs, hidden=None, **kwargs):
    if hidden is None:
        hidden = agent.init_hidden(obs.shape[0], obs.device)
    previous = agent.initial_previous_actions(obs.shape[0], obs.device)
    return agent.step(obs, previous, hidden, **kwargs)


def test_maic_step_shapes() -> None:
    agent = _agent()
    obs = torch.randn(6, 3, 5)
    step = _step(agent, obs)
    assert tuple(step.q.shape) == (6, 3, 4)
    assert tuple(step.q_loc.shape) == (6, 3, 4)
    assert tuple(step.mu.shape) == (6, 3, 3, 4)
    assert tuple(step.alpha.shape) == (6, 3, 3)
    assert tuple(step.hidden.shape) == (6, 3, 16)


def test_maic_alpha_is_a_self_masked_distribution() -> None:
    torch.manual_seed(0)
    agent = _agent()
    step = _step(agent, torch.randn(4, 3, 5))
    assert torch.allclose(step.alpha.sum(dim=-1), torch.ones(4, 3), atol=1e-5)
    assert torch.all(step.alpha.diagonal(dim1=-2, dim2=-1) < 1e-6)
    assert torch.all(step.sigma >= math.sqrt(0.002) - 1e-8)   # var_floor


def test_maic_incentive_bias_shifts_teammate_argmax(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)

    class PushAction2(torch.nn.Module):
        def forward(self, x):
            bias = torch.zeros(*x.shape[:-1], 4)
            bias[..., 2] = 50.0
            return bias

    monkeypatch.setattr(agent.network, "msg_net", PushAction2())
    monkeypatch.setattr(
        agent.network, "attention",
        lambda hidden, z: torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).expand(obs.shape[0], -1, -1),
    )
    step = _step(agent, obs)

    # Receiver 1 gets sender 0's huge bias on action 2; its biased argmax moves there.
    assert torch.all(step.q[:, 1].argmax(dim=-1) == 2)
    # Receivers with no incoming link keep their local Q exactly.
    assert torch.allclose(step.q[:, 0], step.q_loc[:, 0])
    assert torch.allclose(step.q[:, 2], step.q_loc[:, 2])


def test_maic_sparsity_loss_gradient_separation() -> None:
    torch.manual_seed(0)
    agent = _agent()
    step = _step(agent, torch.randn(3, 3, 5))
    agent.sparsity_loss(step).mean().backward()

    for module in (agent.network.w_query, agent.network.w_key):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for module in (agent.network.embed_net, agent.network.msg_net, agent.network.inference_net,
                   agent.network.fc1, agent.network.gru, agent.network.q_head):
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in module.parameters())


def test_maic_teammate_model_loss_gradients() -> None:
    torch.manual_seed(0)
    agent = _agent()
    step = _step(agent, torch.randn(3, 3, 5))
    agent.teammate_model_loss(step, torch.randint(4, (3, 3))).mean().backward()

    for module in (agent.network.embed_net, agent.network.inference_net, agent.network.fc1):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in agent.target_network.parameters())


def test_maic_teammate_model_kl_closed_form(monkeypatch) -> None:
    # teammate_model_loss must equal the hand-computed diagonal-Gaussian KL when the
    # posterior is stubbed to a known distribution.
    torch.manual_seed(0)
    agent = _agent()

    class FixedPosterior(torch.nn.Module):
        def forward(self, x):
            stats = torch.zeros(*x.shape[:-1], 8)   # mu2 = 0, raw var -> exp(0)=1, sigma2 = 1
            return stats

    monkeypatch.setattr(agent.network, "inference_net", FixedPosterior())
    from modmarl.algorithms.maic import MAICStep

    mu = torch.full((2, 3, 3, 4), 0.5)
    sigma = torch.full((2, 3, 3, 4), 1.0)
    step = MAICStep(
        q=torch.randn(2, 3, 4), q_loc=torch.randn(2, 3, 4), mu=mu, sigma=sigma,
        z=mu, alpha=torch.zeros(2, 3, 3), hidden=torch.randn(2, 3, 16),
    )
    loss = agent.teammate_model_loss(step, torch.zeros(2, 3, dtype=torch.long))

    # Per dim: KL(N(0.5, 1) || N(0, 1)) = 0.5^2/2 = 0.125; summed over 4 dims = 0.5.
    assert torch.allclose(loss, torch.full((2,), 0.5), atol=1e-5)


def test_maic_eval_mode_deterministic_and_pruned() -> None:
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    hidden = agent.init_hidden(2, obs.device)

    step_a = _step(agent, obs, hidden, deterministic=True)
    step_b = _step(agent, obs, hidden, deterministic=True)
    assert torch.equal(step_a.q, step_b.q)
    assert torch.equal(step_a.z, step_a.mu)

    sampled_a = _step(agent, obs, hidden)
    sampled_b = _step(agent, obs, hidden)
    assert not torch.equal(sampled_a.q, sampled_b.q)

    pruned = _step(agent, obs, hidden, deterministic=True, prune_threshold=3.0)
    # With delta = 3 and n = 3, every alpha below 1.0 is cut — i.e. all links.
    assert torch.all(pruned.alpha == 0.0)
    assert torch.allclose(pruned.q, pruned.q_loc)


def test_maic_shares_weights_but_is_identity_aware() -> None:
    # The trunk is weight-shared: identical (obs, hidden) rows give identical local Qs.
    # The teammate models are positional (slot j = teammate ID, as in the official
    # one-head encoder), so the network is deliberately NOT permutation-equivariant.
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(2, 3, 5)
    obs[:, 1] = obs[:, 0]
    hidden = torch.randn(2, 3, 16)
    hidden[:, 1] = hidden[:, 0]

    previous = agent.initial_previous_actions(2, obs.device)
    step = agent.step(obs, previous, hidden, deterministic=True)
    assert not torch.allclose(step.q_loc[:, 0], step.q_loc[:, 1])  # identity is an explicit input


def test_maic_train_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "maic.pt"
    summary = train(
        env="navigation",
        n_agents=3,
        horizon=6,
        episodes=6,
        seed=5,
        hidden_dim=16,
        latent_dim=4,
        attention_dim=8,
        mixer_hidden_dim=8,
        buffer_episodes=16,
        batch_episodes=2,
        warmup_episodes=2,
        checkpoint=str(checkpoint),
    )
    assert summary["algorithm"] == "maic"
    assert summary["communication_rate"] == summary["final_evaluation"]["communication_rate"]
    assert checkpoint.exists()
    state_dict = torch.load(str(checkpoint), weights_only=True)
    assert any(key.startswith("network") for key in state_dict)


def test_maic_gaussian_layout_is_all_means_then_all_variances() -> None:
    agent = _agent()
    latent = agent.network.latent_dim
    means = torch.arange(3 * latent, dtype=torch.float32)
    raw_variances = torch.full((3 * latent,), math.log(4.0))
    mu, sigma = agent.network._gaussian(
        torch.cat((means, raw_variances)).unsqueeze(0), modeled_agents=True,
    )
    assert torch.equal(mu.reshape(-1), means)
    assert torch.allclose(sigma, torch.full_like(sigma, 2.0))


def test_maic_vdn_sums_agent_values() -> None:
    agent = _agent(mixer="vdn")
    chosen = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    state = torch.zeros(1, 2, 15)
    assert torch.equal(agent.mix(chosen, state), torch.tensor([[6.0, 15.0]]))


def test_maic_defaults_to_released_two_layer_qmix_hypernetwork() -> None:
    agent = _agent()
    assert agent.mixer_type == "qmix"
    assert isinstance(agent.mixer.hyper_w1, torch.nn.Sequential)
    assert agent.mixer.hyper_w1[0].out_features == 64


def test_maic_previous_action_and_identity_enter_policy_input() -> None:
    torch.manual_seed(3)
    agent = _agent()
    obs = torch.zeros(2, 3, 5)
    hidden = agent.init_hidden(2, obs.device)
    previous_a = agent.initial_previous_actions(2, obs.device)
    previous_b = previous_a.clone()
    previous_b[:, :, 2] = 1.0
    first = agent.step(obs, previous_a, hidden, deterministic=True)
    second = agent.step(obs, previous_b, hidden, deterministic=True)
    assert not torch.allclose(first.q_loc, second.q_loc)
    assert not torch.allclose(first.q_loc[:, 0], first.q_loc[:, 1])


def test_released_hallway_config_feeds_previous_action_and_uses_qmix() -> None:
    # join1.yaml disables obs_last_action only under `env_args`, where Join1Env stores it
    # and never reads it; the controller sees default.yaml's top-level True. The paper
    # agrees ("observation and last action"), so the previous action IS a policy input.
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(train_maic.train).parameters.items()
    }
    assert defaults["mixer"] == "qmix"
    assert defaults["include_previous_action"] is True

    torch.manual_seed(3)
    agent = _agent(include_previous_action=True)
    obs = torch.zeros(2, 3, 5)
    hidden = agent.init_hidden(2, obs.device)
    previous_a = agent.initial_previous_actions(2, obs.device)
    previous_b = torch.nn.functional.one_hot(
        torch.full((2, 3), 2), agent.action_dim,
    ).float()
    first = agent.step(obs, previous_a, hidden, deterministic=True)
    second = agent.step(obs, previous_b, hidden, deterministic=True)
    assert not torch.equal(first.q_loc, second.q_loc)


def test_previous_action_can_be_ablated() -> None:
    torch.manual_seed(3)
    agent = _agent(include_previous_action=False)
    obs = torch.zeros(2, 3, 5)
    hidden = agent.init_hidden(2, obs.device)
    previous_a = agent.initial_previous_actions(2, obs.device)
    previous_b = torch.nn.functional.one_hot(
        torch.full((2, 3), 2), agent.action_dim,
    ).float()
    first = agent.step(obs, previous_a, hidden, deterministic=True)
    second = agent.step(obs, previous_b, hidden, deterministic=True)
    assert torch.equal(first.q_loc, second.q_loc)


def test_maic_incentive_sum_is_over_senders(monkeypatch) -> None:
    # Receiver j's bias must be the COLUMN sum over senders of alpha_ij * v_ij; a
    # silent axis transpose would produce row sums and fail on an asymmetric alpha.
    torch.manual_seed(0)
    agent = _agent()
    obs = torch.randn(1, 3, 5)

    class OnesMsg(torch.nn.Module):
        def forward(self, x):
            return torch.ones(*x.shape[:-1], 4)

    fixed_alpha = torch.tensor([[[0.0, 0.5, 0.5], [0.2, 0.0, 0.6], [0.5, 0.7, 0.0]]])
    monkeypatch.setattr(agent.network, "msg_net", OnesMsg())
    monkeypatch.setattr(agent.network, "attention", lambda hidden, z: fixed_alpha)

    step = _step(agent, obs)
    bias = step.q - step.q_loc
    expected_per_receiver = fixed_alpha.sum(dim=1)   # column sums: (1, 3)
    assert torch.allclose(bias, expected_per_receiver.unsqueeze(-1).expand(-1, -1, 4), atol=1e-5)


def test_maic_mi_posterior_pairing(monkeypatch) -> None:
    torch.manual_seed(0)
    agent = _agent()
    step = _step(agent, torch.randn(1, 3, 5))
    captured = {}

    original = agent.network.inference_net

    class Spy(torch.nn.Module):
        def forward(self, x):
            captured["input"] = x.detach().clone()
            return original(x)

    monkeypatch.setattr(agent.network, "inference_net", Spy())
    executed = torch.tensor([[3, 2, 1]])
    agent.teammate_model_loss(step, executed)

    onehot = torch.nn.functional.one_hot(executed, 4).float()
    for modeler in range(3):
        for modeled in range(3):
            pair = captured["input"][0, modeler, modeled]
            assert torch.allclose(pair[:16], step.hidden[0, modeler])
            assert torch.equal(pair[16:], onehot[0, modeled])


def test_maic_aux_losses_ignore_padding() -> None:
    import numpy as np

    from examples import train_maic
    from modmarl.common.replay import EpisodeReplayBuffer

    def run(garbage: float) -> tuple[float, dict[str, float]]:
        torch.manual_seed(7)
        np.random.seed(7)
        agent = _agent()
        opt = torch.optim.RMSprop(agent.network.parameters(), lr=0.0)
        buffer = EpisodeReplayBuffer(capacity=4, horizon=4, n_agents=3, obs_dim=5)
        rng = np.random.RandomState(0)
        buffer.add_episode(
            obs=rng.randn(3, 3, 5).astype(np.float32),
            actions=rng.randint(0, 4, (2, 3)),
            rewards=rng.randn(2).astype(np.float32),
            dones=np.zeros(2, dtype=np.float32),
        )
        # Poison the padded tail directly; masked losses must not see it.
        buffer.obs[0, 4:] = garbage
        buffer.actions[0, 3:] = 0
        batch = buffer.sample(1, torch.device("cpu"))

        losses = {}
        real_backward = torch.Tensor.backward

        def spy(self, *args, **kwargs):
            losses["total"] = float(self.detach())
            return real_backward(self, *args, **kwargs)

        torch.Tensor.backward = spy
        try:
            metrics = train_maic._update(
                agent, opt, batch, gamma=0.9, mi_loss_weight=0.001, entropy_loss_weight=0.01,
            )
        finally:
            torch.Tensor.backward = real_backward
        return losses["total"], metrics

    clean_loss, metrics = run(0.0)
    poisoned_loss, _ = run(1000.0)
    assert clean_loss == pytest.approx(poisoned_loss, rel=1e-6)
    assert set(metrics) == {
        "td_loss", "teammate_model_loss", "sparsity_loss", "q_mean", "target_mean", "gradient_norm",
    }
    assert all(math.isfinite(value) for value in metrics.values())


def test_maic_collection_acts_on_biased_q(monkeypatch) -> None:
    from examples import train_maic
    from marl_envs import make_env
    from modmarl.common.replay import EpisodeReplayBuffer

    torch.manual_seed(0)
    env = make_env("navigation", 3, 4, 5)
    agent = MAICAgent(n_agents=3, obs_dim=env.obs_dim, action_dim=env.num_actions,
                      hidden_dim=16, latent_dim=4, attention_dim=8, mixer_hidden_dim=8)

    class PushAction2(torch.nn.Module):
        def forward(self, x):
            bias = torch.zeros(*x.shape[:-1], env.num_actions)
            bias[..., 2] = 50.0
            return bias

    monkeypatch.setattr(agent.network, "msg_net", PushAction2())
    monkeypatch.setattr(
        agent.network, "attention",
        lambda hidden, z: (1.0 - torch.eye(3)).unsqueeze(0) / 2.0,
    )
    replay = EpisodeReplayBuffer(capacity=2, horizon=env.horizon, n_agents=3, obs_dim=env.obs_dim)
    train_maic._collect_episode(agent, env, replay, 0.0, env.num_actions, 5, torch.device("cpu"))

    batch = replay.sample(1, torch.device("cpu"))
    # With every receiver's biased Q pushed hard onto action 2, greedy collection
    # must record action 2 everywhere — acting reads the biased Q, not Q_loc.
    assert (batch.actions[batch.mask.bool()] == 2).all()


def test_attention_matches_the_released_equation_two() -> None:
    # Paper Eq. 2 and the release both normalise over receivers m != i, with the sender's
    # own query and the per-pair key, scaled by 1/sqrt(attention_dim). Transcribed here
    # element by element so an axis transpose or a missing self-mask fails loudly.
    torch.manual_seed(43)
    agent = _agent().double()
    network = agent.network
    batch, n = 2, agent.n_agents
    hidden = torch.randn(batch, n, network.hidden_dim, dtype=torch.float64)
    z = torch.randn(batch, n, n, network.latent_dim, dtype=torch.float64)

    got = network.attention(hidden, z)

    scale = math.sqrt(network.attention_dim)
    expected = torch.empty(batch, n, n, dtype=torch.float64)
    for b in range(batch):
        for i in range(n):
            query = network.w_query(hidden[b, i])
            logits = torch.tensor(
                [
                    -math.inf if j == i
                    else float((torch.dot(query, network.w_key(z[b, i, j])) / scale).detach())
                    for j in range(n)
                ],
                dtype=torch.float64,
            )
            expected[b, i] = torch.softmax(logits, dim=-1)

    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-12)
