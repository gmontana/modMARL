"""Seeded learning-regression checks for algorithms with bounded test budgets.

Each test uses the dependency-free validation task declared by that algorithm's
trainer and asserts the mean return over the last 50 episodes beats the first 50
by a calibrated margin. Communication methods whose timing makes the fully
observed navigation task misleading use their explicit partial-observation task.

These are deliberately slow (paper-budget cases can take several minutes) and
excluded from the default suite; run them with
`pytest -o addopts='' -m slow tests/test_learning.py`.
"""

from __future__ import annotations

import importlib
import statistics

import pytest

pytest.importorskip("gymnasium")

from modmarl.algorithms.tarmac import TarMACConfig

CASES = [
    (
        "commnet",
        {
            "env": "navigation",
            "n_agents": 3,
            "horizon": 15,
            "episodes": 100_000,
            "seed": 3,
            "learning_rate": 3e-3,
            "batch_size": 288,
        },
        1.5,
    ),
    (
        "tarmac",
        {
            "env": "target_signaling",
            "n_agents": 3,
            "horizon": 1,
            "episodes": 30_000,
            "seed": 3,
            "config": TarMACConfig(communication_rounds=2),
        },
        0.7,
    ),
    (
        "i2c",
        {
            "env": "paper_i2c_navigation",
            "n_agents": 7,
            "episodes": 5000,
            "horizon": 40,
            "seed": 11,
        },
        8.0,
    ),
    ("schednet", {}, 3.0),
    ("intention_sharing", {}, 3.0),
    ("mdmaddpg", {}, 6.0),
    ("maddpg_m", {"env": None}, 2.0),
    ("maac", {}, 3.0),
    (
        "maddpg",
        {
            "env": "paper_maddpg_navigation",
            "episodes": 5000,
            "seed": 11,
            "evaluation_episodes": 10,
        },
        10.0,
    ),
    (
        "ddpg",
        {
            "env": "noisy_navigation",
            "n_agents": 1,
            "hidden_dims": (64, 48),
            "batch_size": 64,
            "evaluation_episodes": 10,
        },
        2.0,
    ),
    ("vdn", {}, 2.0),
    ("iql", {}, 1.0),
    ("qmix", {}, 2.0),
    ("mappo", {}, 1.0),
    (
        "mat",
        {"episodes": 5000, "seed": 3, "rollout_episodes": 128, "evaluation_episodes": 10},
        3.0,
    ),
    (
        "commformer",
        {"episodes": 5000, "seed": 3, "rollout_episodes": 128, "evaluation_episodes": 10},
        3.0,
    ),
    (
        "iwol",
        {
            "episodes": 1024,
            "seed": 3,
            "mode": "implicit",
            "rollout_episodes": 128,
            "evaluation_episodes": 10,
        },
        3.0,
    ),
    ("ippo", {}, 1.0),
    # The recurrent communication methods are checked on the hidden-gate bridge — the
    # partially observable task their gating/graph machinery is designed for; on the
    # fully observed navigation task communication has nothing to add.
    ("ic3net", {"env": "hidden_gate_bridge_v2", "n_agents": 8, "horizon": 75, "episodes": 5000}, 2.0),
    ("magic", {"env": "hidden_gate_bridge_v2", "n_agents": 8, "horizon": 75, "episodes": 1200}, 2.0),
    ("sms", {"episodes": 800}, 3.0),
]

# Not in this tier: CACOM, CDC, ExpoComm, HAPPO, NDQ, MAIC, and MASIA need larger
# budgets; ATOC and CMVC need a GPU at their curve budgets (ATOC's 2560-sample
# batches take hours on one CPU thread); MARC needs the optional macpp package.
# Their committed three-seed curves provide the learning regression instead.
#
# Run this tier single-threaded (OMP_NUM_THREADS=1): the margins were calibrated
# that way, and CPU reduction order under many threads changes the seeded runs.


def test_learning_cases_only_name_trainable_algorithms() -> None:
    from tools.train_curves import ALGORITHMS

    trainable = {algorithm for algorithm, _ in ALGORITHMS}
    assert {algorithm for algorithm, _, _ in CASES} <= trainable


@pytest.mark.slow
@pytest.mark.parametrize("algorithm,overrides", [(a, o) for a, o, _ in CASES], ids=[a for a, _, _ in CASES])
def test_algorithm_learns_on_navigation(algorithm, overrides) -> None:
    margin = next(m for a, _, m in CASES if a == algorithm)
    kwargs = {"env": "navigation", "episodes": 500, "seed": 0, "checkpoint": None, **overrides}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    trainer = importlib.import_module(f"examples.train_{algorithm}")
    summary = trainer.train(**kwargs)

    returns = summary["returns"]
    early = statistics.mean(returns[:50])
    late = statistics.mean(returns[-50:])
    assert late - early > margin, (
        f"{algorithm}: no learning — first-50 mean {early:.2f}, last-50 mean {late:.2f}"
    )
