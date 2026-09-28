from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from modmarl.common.replay import EpisodeReplayBuffer


def _episode(length: int, n_agents: int = 2, obs_dim: int = 3, terminated: bool = False):
    return {
        "obs": np.arange((length + 1) * n_agents * obs_dim, dtype=np.float32).reshape(length + 1, n_agents, obs_dim),
        "actions": np.ones((length, n_agents), dtype=np.int64),
        "rewards": np.full((length,), 2.0, dtype=np.float32),
        "dones": np.eye(1, length, length - 1, dtype=np.float32).ravel() * float(terminated),
    }


def test_episode_buffer_pads_and_masks_short_episodes() -> None:
    buffer = EpisodeReplayBuffer(capacity=4, horizon=5, n_agents=2, obs_dim=3)
    buffer.add_episode(**_episode(length=3, terminated=True))

    batch = buffer.sample(1, torch.device("cpu"))
    assert tuple(batch.obs.shape) == (1, 6, 2, 3)
    assert tuple(batch.actions.shape) == (1, 5, 2)
    assert torch.equal(batch.mask[0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    # Termination is marked inside the mask; padding is mask-only, not "done".
    assert torch.equal(batch.dones[0], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.all(batch.rewards[0, 3:] == 0.0)
    assert torch.all(batch.obs[0, 4:] == 0.0)


def test_episode_buffer_truncated_episode_has_no_done() -> None:
    buffer = EpisodeReplayBuffer(capacity=4, horizon=3, n_agents=2, obs_dim=3)
    buffer.add_episode(**_episode(length=3, terminated=False))
    batch = buffer.sample(1, torch.device("cpu"))
    assert torch.all(batch.dones[0] == 0.0)
    assert torch.all(batch.mask[0] == 1.0)


def test_episode_buffer_ring_overwrites_at_capacity() -> None:
    buffer = EpisodeReplayBuffer(capacity=2, horizon=4, n_agents=2, obs_dim=3)
    for reward in (1.0, 2.0, 3.0):
        episode = _episode(length=4)
        episode["rewards"][:] = reward
        buffer.add_episode(**episode)
    assert len(buffer) == 2
    rewards = {float(r) for r in buffer.rewards[:, 0]}
    assert rewards == {2.0, 3.0}


def test_episode_buffer_rejects_overlong_episodes() -> None:
    buffer = EpisodeReplayBuffer(capacity=2, horizon=3, n_agents=2, obs_dim=3)
    with pytest.raises(ValueError):
        buffer.add_episode(**_episode(length=4))
