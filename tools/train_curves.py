"""Run the complete algorithm registry and record reproducible learning curves.

Produces one JSON per (algorithm, seed) job — {"algorithm", "seed", "env", "episodes",
"returns"} — for the plotting script. Each registry entry pins its validation task,
budget, and three seeds; the optional ``macpp`` package is required only for MARC.
Source revisions are embedded in every successful output.

Run:
    python tools/train_curves.py --out figures/curve_data --jobs 8 --clear
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for examples.*

from modmarl.algorithms.atoc import ATOCConfig
from modmarl.algorithms.tarmac import TarMACConfig

ALGORITHMS: list[tuple[str, dict]] = [
    (
        "atoc",
        {
            "env": "signed_noisy_navigation",
            "n_agents": 2,
            "episodes": 3_000,
            "updates_per_episode": 1,
            "config": ATOCConfig(actor_learning_rate=1e-4),
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "cacom",
        {
            # At 9k episodes seed 7 sees only three Q-derived gate updates after the
            # released 200k-step forced-link phase; 15k evaluates the learned gate.
            "env": "navigation", "episodes": 15_000, "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "commnet",
        {
            "env": "navigation",
            "n_agents": 3,
            "horizon": 15,
            "episodes": 100_000,
            "learning_rate": 3e-3,
            "batch_size": 288,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "cmvc",
        {
            "env": "paper_navigation",
            "episodes": 1_000,
            "updates_per_episode": 1,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "commformer",
        {
            "env": "navigation",
            "episodes": 5000,
            "rollout_episodes": 128,
            "ppo_epochs": 15,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "iwol",
        {
            "env": "navigation",
            # 78 full release-sized rollouts: seed 3 is still below the deterministic
            # acceptance boundary after 39 rollouts with the released scheduler encoder.
            "episodes": 9984,
            "mode": "implicit",
            "rollout_episodes": 128,
            "ppo_epochs": 15,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "ic3net",
        {
            "env": "hidden_gate_bridge_v2",
            "n_agents": 8,
            "horizon": 75,
            "episodes": 5_000,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "tarmac",
        {
            "env": "target_signaling",
            "n_agents": 3,
            "horizon": 1,
            "episodes": 30_000,
            "config": TarMACConfig(communication_rounds=2),
            "evaluation_episodes": 500,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "i2c",
        {
            "env": "paper_i2c_navigation",
            "episodes": 10_000,
            "horizon": 40,
            "n_agents": 7,
            "evaluation_episodes": 64,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "schednet",
        {
            "env": "navigation",
            # Twenty-five steps per episode reaches the release's 750k-step exploration
            # horizon without changing its epsilon schedule or optimizer protocol.
            "episodes": 30_000,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "intention_sharing",
        {
            "env": "paper_navigation",
            "episodes": 50_000,
            "update_interval": 100,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "magic",
        {
            "env": "hidden_gate_bridge_v2",
            "n_agents": 8,
            "horizon": 75,
            "episodes": 1_200,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "cdc",
        {
            "env": "paper_navigation",
            "horizon": 25,
            "episodes": 5000,
            "variant": "clean",
            "evaluation_episodes": 100,
            "_curve_seeds": [1, 2001, 4001],
        },
    ),
    (
        "mdmaddpg",
        {
            "env": "paper_mdmaddpg_navigation",
            "episodes": 200,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "maddpg_m",
        {
            # The paper trains for 100,000 episodes; this bounded demonstration uses
            # two fifths of that budget while preserving every learner hyperparameter.
            "episodes": 40_000,
            "n_agents": 3,
            "horizon": 25,
            "communication_interval": 5,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    # Seeds pinned: the CLI schedule may differ from published per-algorithm evidence.
    ("marc", {"episodes": 3000, "_curve_seeds": [3, 5, 7]}),   # fails soft without macpp
    (
        "maac",
        {
            "env": "paper_navigation",
            "episodes": 5_000,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "maddpg",
        {
            "env": "paper_maddpg_navigation",
            "episodes": 5000,
            "horizon": 25,
            "batch_size": 1024,
            "update_interval": 100,
            "evaluation_episodes": 100,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "qmix",
        {
            "env": "navigation",
            "episodes": 30_000,
            "warmup_episodes": 32,
            "batch_size": 32,
            "epsilon_anneal_steps": 20_000,
            "evaluation_episodes": 64,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "ndq",
        {
            "env": "ndq_hallway",
            "n_agents": 2,
            "horizon": 16,
            "episodes": 100_000,
            "message_dim": 3,
            "gamma": 0.99,
            "c_beta": 0.1,
            "comm_beta": 1e-2,
            "include_agent_id": True,
            "include_last_action": False,
            "epsilon_anneal_steps": 50_000,
            "update_every_episodes": 16,
            "buffer_episodes": 5000,
            "evaluation_episodes": 300,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "maic",
        {
            "env": "maic_hallway",
            "n_agents": 3,
            "horizon": 20,
            "episodes": 100_000,
            "mixer": "qmix",
            "include_previous_action": True,
            "gamma": 0.99,
            "buffer_episodes": 5000,
            "batch_episodes": 32,
            "epsilon_anneal_steps": 50_000,
            "updates_per_episode": 1,
            "target_update_every": 200,
            "evaluation_episodes": 300,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "masia",
        {
            "env": "ndq_hallway", "n_agents": 2, "horizon": 16,
            "episodes": 20_000, "buffer_episodes": 5000,
            "batch_episodes": 32, "updates_per_episode": 1,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "sms",
        {
            "env": "navigation",
            "episodes": 13_000,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "vdn",
        {
            "env": "navigation",
            "episodes": 20_000,
            "warmup_episodes": 32,
            "batch_size": 32,
            "epsilon_anneal_steps": 50_000,
            "evaluation_episodes": 64,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "iql",
        {
            "env": "navigation",
            "episodes": 5000,
            "learn_start": 1000,
            "epsilon_anneal_steps": 50_000,
            "target_update_interval": 2500,
            "evaluation_episodes": 64,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "mappo",
        {
            "env": "navigation",
            "episodes": 5000,
            "rollout_episodes": 128,
            "ppo_epochs": 10,
            "num_minibatches": 1,
            "chunk_length": 10,
            "evaluation_episodes": 64,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "mat",
        {
            "env": "navigation",
            "episodes": 5000,
            "rollout_episodes": 128,
            "ppo_epochs": 15,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "happo",
        {
            "env": "navigation",
            "episodes": 5000,
            "rollout_episodes": 160,
            "ppo_epochs": 5,
            "evaluation_episodes": 64,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "ippo",
        {
            "env": "navigation",
            "episodes": 15_000,
            "rollout_episodes": 8,
            "ppo_epochs": 4,
            "minibatch_size": 1024,
            "evaluation_episodes": 64,
            "_curve_seeds": [11, 13, 17],
        },
    ),
    (
        "ddpg",
        {
            "env": "noisy_navigation",
            "episodes": 5_000,
            "n_agents": 1,
            "horizon": 25,
            "batch_size": 64,
            "evaluation_episodes": 100,
            "hidden_dims": (400, 300),
            "_curve_seeds": [3, 5, 7],
        },
    ),
    (
        "expocomm",
        {
            "env": "navigation", "n_agents": 3, "episodes": 2000,
            "topology": "one_peer",
            "epsilon_anneal_steps": 5_000,
            "evaluation_episodes": 100,
            "_curve_seeds": [3, 5, 7],
        },
    ),
]

SOURCE_REVISIONS = {
    "atoc": "paper:neurips-2018-atoc;author-code:none",
    "commnet": "3fc1fe801925bac3055d5b4730a7948649eead11",
    "cmvc": "paper:doi-10.1109/TSMC.2025.3604230;repository:GaoZiHong/CMVC-empty",
    "ic3net": "69b7e0ce51a79def593abfef1a976f43e5e13f75",
    "magic": "0ad3a6126f46e475f8d46ab61e67fddbfe99e7d8",
    "tarmac": "paper-only:ICML-2019",
    "i2c": "3f8aab6a69ed2a46236c454bd2336c09ff6ffa36",
    "maddpg_m": "paper-only:arXiv-1812.00922",
    "maddpg": "3ceefa0ada3ff31d633dd0bde8ff95213ce99be3",
    "qmix": "eaf6b063822e2bdf1992d72dc159a500f235dbc4",
    "ndq": "575f2e243bac1a567c072dbea8e093aaa4959511",
    "maic": "2bd47d105ccd64bfba1f1d71981f7723c59ac07f",
    "sms": "e01327924ee197b06fb5012e88ab036e1196e826",
    "vdn": (
        "paper:arXiv-1706.05296v1;"
        "reference:6067b1a85764b4bb4f6ce98e6e128364a95ee403"
    ),
    "iql": "feb0b8acd761eb47cca04f8a80c9c998c34d7a35",
    "mappo": "de66d7a4b23fac2513f56f96f73b3f5cb96695ac",
    "happo": "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2",
    "ippo": "paper-only:arXiv-2011.09533",
    "ddpg": "paper-only:arXiv-1509.02971v6",
    "masia": "0106fea3a31fe29a02d6bdce1e1d3cd453ce04a1",
    "expocomm": "25dc9729c0ac65a283ae76977cfbec6df41249c0",
    "cacom": "97493a0b2c402e88a06d4e0d21327c41bbd21709",
    "marc": "43b71357bce11b075e799876e8c4f1deade787d8",
    "mat": "be3ff49c8264d454c1fe2c41582aa2bfc98498c8",
    "commformer": "c6cd65ea0b902703284cb031b7df732c0ff8efa5",
    "iwol": "de3bc5b1e50bd9c4d90672a6355269ea2917fd28",
    "schednet": "ffa03007cc654000a859856401231a986a01fbd0",
    "intention_sharing": "paper:qpsl2dR9twy;reference:e5d4527d7f7fecf36a1e9fe60969352eb07cd691",
    "cdc": "46d287376c31cb183006b01542411d97cf95679a",
    "maac": "6174a01251251e6778c4ada26bc8d9cd930e3856",
    "mdmaddpg": "20dffc4e12b5b490ad30defb49887d2599aeece6",
}


def _run_job(job: tuple[str, dict, int, int, str]) -> str:
    algorithm, kwargs, seed, episodes, out_dir = job
    import torch

    torch.set_num_threads(1)               # jobs run in parallel; avoid oversubscription
    out_path = Path(out_dir) / f"{algorithm}_seed{seed}.json"
    started = time.time()
    try:
        module = importlib.import_module(f"examples.train_{algorithm}")
        run_kwargs = {"episodes": episodes, "seed": seed, "checkpoint": None, **kwargs}
        summary = module.train(**run_kwargs)
        payload = {
            "algorithm": algorithm,
            "seed": seed,
            "env": summary["env"],
            "episodes": summary["episodes"],
            "config": summary.get(
                "config", {key: value for key, value in run_kwargs.items() if key != "checkpoint"},
            ),
            "source_revision": SOURCE_REVISIONS.get(algorithm),
            "returns": summary["returns"],
            "minutes": round((time.time() - started) / 60.0, 2),
        }
        for key in (
            "evaluation_seeds", "initial_evaluation", "random_evaluation", "final_evaluation",
            "message_ablated_evaluation", "validation_criterion", "teacher_episodes",
            "teacher_returns", "prior_examples", "prior_positive_rate", "prior_loss",
            "influence_threshold",
            "training_metrics",
            "evaluation_returns", "communication_rate", "total_steps", "variant",
            "training_actor_communication_rate", "training_critic_communication_rate",
        ):
            if key in summary:
                payload[key] = summary[key]
        out_path.write_text(json.dumps(payload))
        return f"done {algorithm} seed {seed} ({payload['minutes']} min)"
    except Exception:
        out_path.with_suffix(".error").write_text(traceback.format_exc())
        return f"FAILED {algorithm} seed {seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record learning curves for every algorithm.")
    parser.add_argument("--out", default="curves")
    parser.add_argument("--seeds", type=int, default=3, help="use seeds 0..N-1")
    parser.add_argument(
        "--seed-values",
        nargs="+",
        type=int,
        default=None,
        help="explicit seeds; overrides --seeds and makes published runs reproducible",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these algorithms")
    parser.add_argument("--env", default=None, help="override the task (skips algorithms tied to their own env)")
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="remove existing JSON/error files for selected algorithms before running",
    )
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    selected = [(a, k) for a, k in ALGORITHMS if args.only is None or a in args.only]
    # Comparability: every navigation job runs the same team size and horizon regardless
    # of its trainer's own defaults (CDC's example defaults to horizon 50, for instance).
    selected = [
        (a, {"horizon": 25, "n_agents": 3, **k} if k.get("env") == "navigation" else k)
        for a, k in selected
    ]
    if args.env is not None:
        selected = [(a, {**k, "env": args.env}) for a, k in selected if "env" in k]
    if args.n_agents is not None:
        selected = [(a, {**k, "n_agents": args.n_agents}) for a, k in selected]
    if args.horizon is not None:
        selected = [(a, {**k, "horizon": args.horizon}) for a, k in selected]
    if args.clear:
        _clear_outputs(Path(args.out), [algorithm for algorithm, _ in selected])
    seeds = _selected_seeds(args.seeds, args.seed_values)
    jobs = []
    for algorithm, kwargs in selected:
        run_kwargs = dict(kwargs)
        algorithm_seeds = run_kwargs.pop("_curve_seeds", seeds)
        jobs.extend(
            (algorithm, run_kwargs, seed, args.episodes, args.out)
            for seed in algorithm_seeds
        )
    print(f"{len(jobs)} jobs on {args.jobs} workers -> {args.out}", flush=True)
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for message in pool.imap_unordered(_run_job, jobs):
            print(message, flush=True)


def _selected_seeds(count: int, explicit: list[int] | None) -> list[int]:
    """Return the exact seed schedule recorded in each curve JSON file."""
    return list(explicit) if explicit is not None else list(range(count))


def _clear_outputs(out_dir: Path, algorithms: list[str]) -> None:
    """Remove stale results only for algorithms selected for regeneration."""
    for algorithm in algorithms:
        for suffix in ("json", "error"):
            for path in out_dir.glob(f"{algorithm}_seed*.{suffix}"):
                path.unlink()


if __name__ == "__main__":
    main()
