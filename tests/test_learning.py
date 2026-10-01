"""Optional learning checks: frozen acceptance protocols and legacy probes.

Frozen recipes run all registered seeds through the same provenance-aware CLI
used to create the committed evidence. They compare fixed evaluation policies,
not changes in exploratory training returns. The remaining legacy probes retain
older first/last-return checks; those alone are not release acceptance evidence.
Run explicitly with ``pytest -o addopts='' -m slow tests/test_learning.py``.
"""

from __future__ import annotations

import importlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

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
    ("sms", {"episodes": 800}, 3.0),
]

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = sorted((ROOT / "validation/recipes").glob("*.json"))
CONFIRMATION_CASES = [
    (path, seed) for path in PROTOCOLS for seed in json.loads(path.read_text())["seeds"]
]

# Run these optional checks single-threaded. Most recipes take minutes per seed;
# MAIC's larger budget takes substantially longer. CI checks committed evidence
# and mechanisms instead of retraining the catalogue on every change.


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


def test_frozen_learning_recipes_have_complete_preregistered_rules() -> None:
    from tools.train_curves import SOURCE_REVISIONS

    legacy = {name for name, _, _ in CASES}
    for path in PROTOCOLS:
        protocol = json.loads(path.read_text())
        assert protocol["algorithm"] == path.stem
        assert protocol["algorithm"] in SOURCE_REVISIONS
        assert protocol["algorithm"] not in legacy
        assert len(protocol["seeds"]) >= 3
        assert len(set(protocol["seeds"])) == len(protocol["seeds"])
        assert protocol["criteria"]["return_margin_over_initial"] > 0
        assert protocol["criteria"]["return_margin_over_random"] > 0
        assert set(protocol["criteria"]) - {
            "scope", "return_margin_over_initial", "return_margin_over_random",
        }


@pytest.mark.slow
@pytest.mark.parametrize(
    "protocol,seed", CONFIRMATION_CASES,
    ids=[f"{path.stem}-seed{seed}" for path, seed in CONFIRMATION_CASES],
)
def test_frozen_learning_recipe(protocol, seed, tmp_path) -> None:
    subprocess.run(
        [sys.executable, "-m", "tools.run_validation", "--protocol", str(protocol),
         "--seed", str(seed), "--out", str(tmp_path / "run")],
        cwd=ROOT, check=True,
    )
