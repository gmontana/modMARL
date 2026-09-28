"""Regression tests for done semantics at replay-add time.

Time-limit truncation is not termination: the TD bootstrap must survive the
horizon boundary, so every off-policy trainer stores ``done=terminated`` only
(truncation just ends the rollout loop; MAPPO instead bootstraps V(s_T) in its
GAE computation). The built-in ``navigation`` and ``noisy_navigation`` envs
never terminate, so every ``done`` recorded during training must be False.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from modmarl.algorithms import cmvc as cmvc_module
from modmarl.algorithms import ddpg as ddpg_module
from modmarl.algorithms import i2c as i2c_module
from modmarl.algorithms import intention_sharing as intention_sharing_module
from modmarl.algorithms import maac as maac_module
from modmarl.algorithms import maddpg as maddpg_module
from modmarl.algorithms import mdmaddpg as mdmaddpg_module
from modmarl.algorithms import qmix as qmix_module
from modmarl.algorithms.atoc import algorithm as atoc_module
from modmarl.common import replay as replay_module

_BASE = {
    "env": "navigation",
    "n_agents": 3,
    "horizon": 6,
    "episodes": 2,
    "seed": 5,
    "hidden_dim": 32,
    "buffer_size": 64,
    "batch_size": 4,
    "warmup_steps": 0,
}

TRAINERS = [
    ("train_atoc", {}),
    ("train_cmvc", {}),
    ("train_ddpg", {"env": "noisy_navigation"}),
    ("train_maddpg", {}),
    ("train_cdc", {"message_dim": 16, "diffusion_steps": 6, "diffusion_max": 2.0}),
    ("train_maac", {"attend_heads": 4}),
    ("train_qmix", {"mixer_hidden_dim": 8}),
    ("train_i2c", {"message_dim": 16}),
    ("train_intention_sharing", {"message_dim": 16}),
    ("train_mdmaddpg", {"n_agents": 2, "memory_dim": 16, "hidden_dim": None}),
    ("train_schednet", {"message_dim": 2}),
    ("train_maddpg_m", {"env": None, "n_agents": 2}),
]


@pytest.fixture
def captured_dones(monkeypatch):
    captured: list[bool] = []
    for cls_name in ("ReplayBuffer", "MARCReplayBuffer", "MemoryReplayBuffer", "MADDPGMReplayBuffer"):
        cls = getattr(replay_module, cls_name)
        original = cls.add

        def wrapper(self, *args, _original=original, **kwargs):
            # done is the 5th positional arg of every buffer's add(obs, actions,
            # reward/…, next_obs, done, …) or an explicit keyword.
            done = kwargs["done"] if "done" in kwargs else args[4]
            captured.append(bool(done))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "add", wrapper)
    original_add_episode = qmix_module.QMIXReplayBuffer.add_episode

    original_episode_add = replay_module.EpisodeReplayBuffer.add_episode

    def episode_wrapper(self, *args, **kwargs):
        dones = kwargs["dones"]
        captured.extend(bool(done) for done in dones)
        return original_episode_add(self, *args, **kwargs)

    monkeypatch.setattr(replay_module.EpisodeReplayBuffer, "add_episode", episode_wrapper)

    original_atoc_add = atoc_module.ATOCReplayBuffer.add

    def atoc_wrapper(self, *args, **kwargs):
        dones = kwargs["dones"] if "dones" in kwargs else args[4]
        captured.extend(bool(done) for done in np.asarray(dones).reshape(-1))
        return original_atoc_add(self, *args, **kwargs)

    monkeypatch.setattr(atoc_module.ATOCReplayBuffer, "add", atoc_wrapper)

    original_ddpg_add = ddpg_module.DDPGReplayBuffer.add

    def ddpg_wrapper(self, *args, **kwargs):
        done = kwargs["done"] if "done" in kwargs else args[4]
        captured.append(bool(done))
        return original_ddpg_add(self, *args, **kwargs)

    monkeypatch.setattr(ddpg_module.DDPGReplayBuffer, "add", ddpg_wrapper)

    original_maddpg_add = maddpg_module.MADDPGReplayBuffer.add

    def maddpg_wrapper(self, *args, **kwargs):
        dones = kwargs["dones"] if "dones" in kwargs else args[4]
        captured.extend(bool(done) for done in np.asarray(dones).reshape(-1))
        return original_maddpg_add(self, *args, **kwargs)

    monkeypatch.setattr(maddpg_module.MADDPGReplayBuffer, "add", maddpg_wrapper)

    original_maac_add = maac_module._MAACReplayBuffer.add

    def maac_wrapper(self, *args, **kwargs):
        dones = kwargs["dones"] if "dones" in kwargs else args[4]
        captured.extend(bool(done) for done in np.asarray(dones).reshape(-1))
        return original_maac_add(self, *args, **kwargs)

    monkeypatch.setattr(maac_module._MAACReplayBuffer, "add", maac_wrapper)

    original_mdmaddpg_add = mdmaddpg_module._MDReplayBuffer.add

    def mdmaddpg_wrapper(self, *args, **kwargs):
        dones = kwargs["dones"] if "dones" in kwargs else args[4]
        captured.extend(bool(done) for done in np.asarray(dones).reshape(-1))
        return original_mdmaddpg_add(self, *args, **kwargs)

    monkeypatch.setattr(mdmaddpg_module._MDReplayBuffer, "add", mdmaddpg_wrapper)

    original_intention_add = intention_sharing_module._ReplayBuffer.add

    def intention_wrapper(self, *args, **kwargs):
        dones = kwargs["dones"] if "dones" in kwargs else args[4]
        captured.extend(bool(done) for done in np.asarray(dones).reshape(-1))
        return original_intention_add(self, *args, **kwargs)

    monkeypatch.setattr(intention_sharing_module._ReplayBuffer, "add", intention_wrapper)

    original_i2c_add = i2c_module.I2CReplayBuffer.add

    def i2c_wrapper(self, *args, **kwargs):
        done = kwargs["done"] if "done" in kwargs else args[4]
        captured.append(bool(done))
        return original_i2c_add(self, *args, **kwargs)

    monkeypatch.setattr(i2c_module.I2CReplayBuffer, "add", i2c_wrapper)

    def qmix_wrapper(self, *args, **kwargs):
        dones = kwargs["dones"]
        captured.extend(bool(done) for done in dones)
        return original_add_episode(self, *args, **kwargs)

    monkeypatch.setattr(qmix_module.QMIXReplayBuffer, "add_episode", qmix_wrapper)
    return captured


@pytest.mark.parametrize("module_name,overrides", TRAINERS)
def test_offpolicy_trainer_stores_termination_only(module_name, overrides, captured_dones):
    trainer = importlib.import_module(f"examples.{module_name}")
    kwargs = {**_BASE, **overrides}
    if module_name == "train_atoc":
        kwargs = {
            "env": "paper_navigation",
            "episodes": 2,
            "seed": 5,
            "config": atoc_module.ATOCConfig(
                actor_hidden_dims=(16, 8, 8, 4),
                critic_hidden_dims=(16, 8),
                attention_hidden_dim=4,
                channel_hidden_dim=4,
                batch_size=4,
                replay_capacity=64,
                warmup_episodes=0,
            ),
            "updates_per_episode": 0,
            "evaluation_episodes": 1,
        }
    if module_name == "train_cmvc":
        kwargs = {
            "env": "navigation",
            "n_agents": 3,
            "horizon": 6,
            "episodes": 2,
            "seed": 5,
            "config": cmvc_module.CMVCConfig(
                actor_hidden_dim=8,
                critic_hidden_dim=16,
                hyper_hidden_dim=8,
                message_hidden_dim=6,
                batch_size=2,
                replay_capacity=8,
            ),
            "updates_per_episode": 0,
            "evaluation_episodes": 1,
        }
    if module_name == "train_ddpg":
        kwargs["hidden_dims"] = (32, 24)
        kwargs["evaluation_episodes"] = 1
        kwargs.pop("hidden_dim")
        kwargs.pop("warmup_steps")
        kwargs.pop("update_interval", None)
    if module_name == "train_maac":
        kwargs["evaluation_episodes"] = 1
        kwargs.pop("warmup_steps")
        kwargs.pop("update_interval", None)
    if module_name == "train_mdmaddpg":
        kwargs["evaluation_episodes"] = 1
        kwargs.pop("warmup_steps")
        kwargs.pop("update_interval", None)
    if module_name == "train_qmix":
        kwargs["warmup_episodes"] = kwargs.pop("warmup_steps")
        kwargs["evaluation_episodes"] = 1
    if module_name in {"train_cdc", "train_intention_sharing", "train_schednet"}:
        kwargs["evaluation_episodes"] = 1
    if module_name == "train_schednet":
        kwargs.pop("hidden_dim")
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    trainer.train(**kwargs)
    assert captured_dones, "no transitions were recorded"
    assert not any(captured_dones), "truncation must not be stored as termination"
