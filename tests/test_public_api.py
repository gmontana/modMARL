"""Public-API contract + cross-paradigm reproducibility.

Guards two things the algorithm surface can silently break: every algorithm must stay
exported from the top-level package and expose an importable trainer, and training must be
reproducible from a fixed seed (one representative per paradigm: off-policy, value-based, on-policy).
"""

from __future__ import annotations

import importlib
import math

import pytest

pytest.importorskip("gymnasium")

import modmarl

# (example module suffix, primary public class exported from modmarl)
ALGORITHMS = [
    ("atoc", "ATOCLearner"),
    ("cacom", "CACOMAgent"),
    ("cmvc", "CMVCLearner"),
    ("commnet", "CommNetAgent"),
    ("commformer", "CommFormerAgent"),
    ("cdc", "CDCPolicy"),
    ("ddpg", "DDPGAgent"),
    ("expocomm", "ExpoCommAgent"),
    ("i2c", "I2CAgent"),
    ("ic3net", "IC3NetAgent"),
    ("intention_sharing", "IntentionSharingLearner"),
    ("happo", "HAPPOAgent"),
    ("ippo", "IPPOAgent"),
    ("iql", "IQLAgent"),
    ("iwol", "IWoLAgent"),
    ("maac", "MAACAgent"),
    ("maddpg", "MADDPGAgent"),
    ("magic", "MAGICAgent"),
    ("maic", "MAICAgent"),
    ("maddpg_m", "MADDPGMAgent"),
    ("mappo", "MAPPOAgent"),
    ("mat", "MATAgent"),
    ("marc", "MARCAgent"),
    ("masia", "MASIAAgent"),
    ("mdmaddpg", "MDMADDPGAgent"),
    ("ndq", "NDQAgent"),
    ("qmix", "QMIXAgent"),
    ("vdn", "VDNAgent"),
    ("schednet", "SchedNetAgent"),
    ("sms", "SMSAgent"),
    ("tarmac", "TarMACAgent"),
]


@pytest.mark.parametrize("module,cls", ALGORITHMS)
def test_algorithm_exported_from_top_level(module, cls):
    assert cls in modmarl.__all__, f"{cls} missing from modmarl.__all__"
    assert hasattr(modmarl, cls), f"{cls} not importable from modmarl"


@pytest.mark.parametrize("module,cls", ALGORITHMS)
def test_algorithm_module_exports_are_curated(module, cls):
    algorithm_module = importlib.import_module(f"modmarl.algorithms.{module}")
    assert hasattr(algorithm_module, "__all__"), f"modmarl.algorithms.{module} missing __all__"
    assert cls in algorithm_module.__all__, f"{cls} missing from modmarl.algorithms.{module}.__all__"
    for name in algorithm_module.__all__:
        assert hasattr(algorithm_module, name), f"{name} listed in {module}.__all__ but not defined"


@pytest.mark.parametrize("module,cls", ALGORITHMS)
def test_example_trainer_importable(module, cls):
    trainer = importlib.import_module(f"examples.train_{module}")
    assert callable(trainer.train), f"examples.train_{module}.train is not callable"


def _assert_reproducible(train, kwargs):
    first = train(**kwargs)
    second = train(**kwargs)
    assert math.isfinite(first["final_return"])
    assert first["final_return"] == second["final_return"], "training is not seed-reproducible"


def test_ddpg_reproducible():
    from examples.train_ddpg import train

    _assert_reproducible(
        train,
        dict(
            env="noisy_navigation",
            n_agents=3,
            horizon=6,
            episodes=2,
            seed=5,
            hidden_dims=(32, 24),
            buffer_size=64,
            batch_size=4,
            evaluation_episodes=1,
        ),
    )


def test_qmix_reproducible():
    from examples.train_qmix import train

    _assert_reproducible(train, dict(env="navigation", n_agents=3, horizon=6, episodes=2, seed=5,
                                     hidden_dim=32, mixer_hidden_dim=16, buffer_size=64, batch_size=2,
                                     warmup_episodes=2, evaluation_episodes=1))


def test_mappo_reproducible():
    from examples.train_mappo import train

    _assert_reproducible(train, dict(env="navigation", n_agents=3, horizon=6, episodes=4, seed=5,
                                     hidden_dim=32, rollout_episodes=2, num_minibatches=1,
                                     chunk_length=3, evaluation_episodes=1, ppo_epochs=2))
