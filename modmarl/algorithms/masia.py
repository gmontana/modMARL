"""MASIA implementation.

Original paper:
Cong Guan, Feng Chen, Lei Yuan, Chenghe Wang, Hao Yin, Zongzhang Zhang, Yang Yu.
"Efficient Multi-agent Communication via Self-supervised Information Aggregation."
Advances in Neural Information Processing Systems (NeurIPS), 2022. Cite the NeurIPS
proceedings: arXiv 2302.09605 is the *extended journal version* under a different
title ("... for Online and Offline Multi-agent Reinforcement Learning"), used here
only for its Appendix B integration-network and hyperparameter details.
Official code: https://github.com/chenf-ai/MASIA (Apache-2.0), reference commit
0106fea3a31fe29a02d6bdce1e1d3cd453ce04a1.

QMIX where agents broadcast their observations and a shared Information Aggregation
Encoder compresses all n broadcasts into one representation z: scaled dot-product
self-attention over the messages (paper Eq. 1-2), a shared per-agent GRUCell
integration network (design (c) of the extended version's Appendix B.1 — the variant
all reported results use), and a per-agent linear bottleneck whose n codes are
flattened, so permuting the agents permutes the z_slot_dim-sized slots of z. Every
agent receives the same z and extracts its own view with a focusing gate,
w_i = sigmoid(W_g o_i) and z_bar_i = w_i * z, feeding concat(Linear(o_i), z_bar_i)
into the standard recurrent Q-network mixed by QMIX. On top of the TD loss (which
deliberately reaches the encoder), the encoder is trained self-supervised: a decoder
reconstructs the global state from z (Eq. 3), and an SPR-style multi-step objective
rolls a residual latent transition model K steps forward, regressing projected +
predicted latents onto projections of a hard-copied target encoder's z (Eq. 4-7),
alongside a reward-prediction head on the same rolled-out latents. The focusing gate
is shaped only by the TD signal.

The released controller contract is preserved: the encoder and focusing gate see
local observation plus optional previous action and agent ID; the local observation
embedding sees only the environment observation, and the extra inputs are appended
after focusing. Previous action is explicit and defaults off for the paper Hallway
configuration. The global state is the concatenation of observations
(house convention; official uses the env state), so the reconstruction autoencodes
concat-obs through the per-agent bottleneck — still a real n*obs_dim -> n*z_slot_dim
compression. Only the shipped-default `ob_attn_ae` encoder is included — the skip-connection,
VAE and noisy-channel variants are omitted; (4) QMIX mixing only (the official QPLEX
learner variant is skipped; masia_vdn merely swaps in VDN).
As in the release, the online encoder has distinct TD-target and momentum copies: the
TD target copy feeds target utilities, while the momentum copy and projection define
gradient-free SPR targets. Kept exactly as the official code has them: the k=0 SPR
alignment term (absent from paper Eq. 4), double-Q targets, the residual
transition model, and hard copies (momentum_tau = 1) of encoder and projection at
every target update. Available-action masking is part of the released learner but has
no counterpart in modMARL's environments, so it is not modelled.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from ..common.nn import build_mlp
from .qmix import QMixer


class InformationAggregationEncoder(nn.Module):
    """MASIA's IAE: self-attention over all agents' broadcast observations, shared
    per-agent GRU integration, and a per-agent bottleneck; plus the state decoder.

    encode: obs (B, n, obs_dim), enc_hidden (B, n, enc_hidden_dim)
        -> z (B, n * z_slot_dim), enc_hidden'. Row i of the attention output is
        agent i's querying result over all messages; slot i of z is a set function
        of the other agents' messages, so permuting agents permutes the slots.
    decode: z -> (B, n * obs_dim), the global state (concatenated observations).
    """

    def __init__(
        self, n_agents: int, obs_dim: int, attn_dim: int = 16, enc_hidden_dim: int = 32,
        z_slot_dim: int = 8, state_obs_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.attn_dim = attn_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.query = nn.Linear(obs_dim, attn_dim)
        self.key = nn.Linear(obs_dim, attn_dim)
        self.value = nn.Linear(obs_dim, enc_hidden_dim)
        self.gru = nn.GRUCell(enc_hidden_dim, enc_hidden_dim)   # shared integration cell, design (c)
        self.bottleneck = nn.Linear(enc_hidden_dim, z_slot_dim)
        self.decoder = build_mlp(
            n_agents * z_slot_dim, [], n_agents * (state_obs_dim or obs_dim),
        )

    def encode(self, obs: Tensor, enc_hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch, n, _ = obs.shape
        scores = self.query(obs) @ self.key(obs).transpose(-2, -1) / math.sqrt(self.attn_dim)
        attended = torch.softmax(scores, dim=-1) @ self.value(obs)          # (B, n, enc_hidden)
        new_hidden = self.gru(
            attended.reshape(batch * n, self.enc_hidden_dim),
            enc_hidden.reshape(batch * n, self.enc_hidden_dim),
        ).view(batch, n, self.enc_hidden_dim)
        return self.bottleneck(new_hidden).reshape(batch, -1), new_hidden

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class MASIATransitionModel(nn.Module):
    """Latent forward model h_psi with a residual output (paper Eq. 5-6), plus the
    reward head enabled in the official online config.

    forward: z (..., n * z_slot_dim), actions_onehot (..., n, action_dim) -> z_next like z.
    predict_reward: z -> (..., 1).
    """

    def __init__(
        self, n_agents: int, action_dim: int, z_dim: int, action_embed_dim: int = 8, hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_embed = nn.Linear(action_dim, action_embed_dim)
        self.joint_action_mlp = build_mlp(n_agents * action_embed_dim, [hidden_dim], hidden_dim)
        self.z_mlp = build_mlp(z_dim, [hidden_dim], hidden_dim)
        self.delta_mlp = build_mlp(2 * hidden_dim, [hidden_dim], z_dim)
        self.reward_head = build_mlp(z_dim, [hidden_dim], 1)

    def forward(self, z: Tensor, actions_onehot: Tensor) -> Tensor:
        joint = torch.relu(self.action_embed(actions_onehot)).flatten(start_dim=-2)
        x = torch.cat([self.z_mlp(z), self.joint_action_mlp(joint)], dim=-1)
        return z + self.delta_mlp(x)                                        # residual: z' = z + Delta

    def predict_reward(self, z: Tensor) -> Tensor:
        return self.reward_head(z)


class _FocusingQNetwork(nn.Module):
    """Focusing gate + PyMARL-shape GRU Q head, shared across agents.

    Every agent receives the same z; the gate w_i = sigmoid(W_g o_i) weighs each
    agent's relevant dimensions (z_bar_i = w_i * z) before the recurrent Q input
    concat(Linear(o_i), z_bar_i).
    """

    def __init__(
        self, obs_dim: int, input_dim: int, action_dim: int, z_dim: int,
        ob_embed_dim: int, hidden_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.gate = nn.Linear(input_dim, z_dim)
        self.ob_fc = nn.Linear(obs_dim, ob_embed_dim)
        self.fc1 = nn.Linear(ob_embed_dim + z_dim + input_dim - obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, inputs: Tensor, z: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        # obs (B, n, obs_dim), z (B, n*z_slot_dim), hidden (B, n, hidden) -> (q, hidden')
        batch, n, _ = inputs.shape
        obs, extras = inputs[..., : self.obs_dim], inputs[..., self.obs_dim :]
        gated = torch.sigmoid(self.gate(inputs)) * z.unsqueeze(1)
        x = torch.cat([self.ob_fc(obs), gated, extras], dim=-1)
        x = torch.relu(self.fc1(x)).reshape(batch * n, self.hidden_dim)
        new_hidden = self.gru(x, hidden.reshape(batch * n, self.hidden_dim)).view(batch, n, self.hidden_dim)
        return self.head(new_hidden), new_hidden


class MASIAAgent(nn.Module):
    """MASIA: aggregation encoder + focusing recurrent Q-network + QMIX mixer, with
    target copies of encoder, projection, Q-network and mixer.

    Two recurrent streams must both be threaded through time and replayed from
    zeros: the Q-network GRU hidden (B, n, hidden_dim) and the encoder integration
    GRU hidden (B, n, enc_hidden_dim); ``init_state`` returns them in that order.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        enc_hidden_dim: int = 32,
        attn_dim: int = 16,
        z_slot_dim: int = 8,
        ob_embed_dim: int = 32,
        spr_dim: int = 32,
        action_embed_dim: int = 8,
        model_hidden_dim: int = 64,
        mixer_hidden_dim: int = 32,
        mixer: str = "qmix",
        include_previous_action: bool = False,
        include_agent_id: bool = True,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.action_dim = action_dim
        self.include_previous_action = include_previous_action
        self.include_agent_id = include_agent_id
        input_dim = obs_dim + action_dim * include_previous_action + n_agents * include_agent_id
        z_dim = n_agents * z_slot_dim
        self.encoder = InformationAggregationEncoder(
            n_agents, input_dim, attn_dim, enc_hidden_dim, z_slot_dim,
            state_obs_dim=obs_dim,
        )
        self.q_network = _FocusingQNetwork(
            obs_dim, input_dim, action_dim, z_dim, ob_embed_dim, hidden_dim,
        )
        # SPR heads: online projection + predictor; the target side has no predictor.
        self.projection = build_mlp(z_dim, [64], spr_dim)
        self.predictor = build_mlp(spr_dim, [64], spr_dim)
        self.transition_model = MASIATransitionModel(n_agents, action_dim, z_dim, action_embed_dim, model_hidden_dim)
        if mixer == "qmix":
            self.mixer = QMixer(
                n_agents, n_agents * obs_dim, mixer_hidden_dim,
                hypernet_hidden_dim=64,
            )
        elif mixer == "vdn":
            self.mixer = _VDNMixer()
        else:
            raise ValueError("mixer must be 'qmix' or 'vdn'")
        self.mixer_name = mixer
        self.target_encoder = copy.deepcopy(self.encoder)
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projection = copy.deepcopy(self.projection)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_mixer = copy.deepcopy(self.mixer)

    def init_state(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros(batch, self.n_agents, self.hidden_dim, device=device),
            torch.zeros(batch, self.n_agents, self.enc_hidden_dim, device=device),
        )

    def build_inputs(self, obs: Tensor, previous_actions: Tensor | None = None) -> Tensor:
        """Build released controller inputs from ``obs: (B, N, O)``."""
        inputs = [obs]
        if self.include_previous_action:
            if previous_actions is None:
                previous_actions = obs.new_zeros(*obs.shape[:2], self.action_dim)
            inputs.append(previous_actions)
        if self.include_agent_id:
            identities = torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype)
            inputs.append(identities.unsqueeze(0).expand(obs.shape[0], -1, -1))
        return torch.cat(inputs, dim=-1)

    def forward(
        self, obs: Tensor, q_hidden: Tensor, enc_hidden: Tensor,
        previous_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Acting path: aggregate all broadcast observations, then the gated Q step."""
        inputs = self.build_inputs(obs, previous_actions)
        z, enc_hidden = self.enc_forward(inputs, enc_hidden)
        q, q_hidden = self.q_forward(inputs, z, q_hidden)
        return q, q_hidden, enc_hidden

    def enc_forward(self, obs: Tensor, enc_hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Online encoder only: (B, n, obs_dim) -> shared z (B, n*z_slot_dim), enc_hidden'."""
        inputs = self.build_inputs(obs) if obs.shape[-1] == self.obs_dim else obs
        return self.encoder.encode(inputs, enc_hidden)

    def q_forward(self, obs: Tensor, z: Tensor, q_hidden: Tensor, *, target: bool = False) -> tuple[Tensor, Tensor]:
        network = self.target_q_network if target else self.q_network
        inputs = self.build_inputs(obs) if obs.shape[-1] == self.obs_dim else obs
        return network(inputs, z, q_hidden)

    @torch.no_grad()
    def target_enc_forward(self, obs: Tensor, enc_hidden: Tensor) -> tuple[Tensor, Tensor]:
        """TD-target encoder path, distinct from the SPR momentum encoder."""
        inputs = self.build_inputs(obs) if obs.shape[-1] == self.obs_dim else obs
        return self.target_encoder.encode(inputs, enc_hidden)

    def project(self, z: Tensor) -> Tensor:
        """Online SPR branch: projection then predictor (the BYOL-style asymmetry)."""
        return self.predictor(self.projection(z))

    @torch.no_grad()
    def target_project_enc(self, obs: Tensor, enc_hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Target path: target encoder then target projection, gradient-free.

        Returns (projected (B, spr_dim), z (B, n*z_slot_dim), enc_hidden') so one
        target-encoder unroll feeds both the SPR targets and the target Q input.
        """
        inputs = self.build_inputs(obs)
        z, enc_hidden = self.momentum_encoder.encode(inputs, enc_hidden)
        return self.momentum_projection(z), z, enc_hidden

    def update_targets(self) -> None:
        """Hard-copy the online networks into the target networks (momentum_tau = 1)."""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.momentum_encoder.load_state_dict(self.encoder.state_dict())
        self.momentum_projection.load_state_dict(self.projection.state_dict())
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    @property
    def target_projection(self) -> nn.Module:
        """The released name for the SPR momentum projection."""
        return self.momentum_projection


class _VDNMixer(nn.Module):
    """VDN's parameter-free sum with the same call contract as QMIX."""

    def forward(self, agent_qs: Tensor, state: Tensor) -> Tensor:
        del state
        return agent_qs.sum(dim=-1)


__all__ = ["InformationAggregationEncoder", "MASIAAgent", "MASIATransitionModel"]
