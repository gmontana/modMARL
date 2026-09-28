"""A small, reproducible signaling example; no general experiment framework.

Run ``python -m modmarl.demo --help``. The two training recipes retain their own
updates. This module owns only the fixed task, artifacts, and user commands.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

import marl_envs
from marl_envs import make_env
from modmarl.algorithms.ippo import IPPOAgent
from modmarl.algorithms.tarmac import TarMACAgent, TarMACConfig
from modmarl.common.provenance import utc_now_iso, write_json_with_provenance
from modmarl.training import ippo, tarmac

TRAINING_SEEDS = (101, 103, 107)
EVALUATION_SEED = 100_000
EVALUATION_EPISODES = 500
LEARNING_EPISODES = 30_000
RECIPES = {"tarmac": tarmac, "ippo": ippo}


def source_hashes() -> dict[str, str]:
    """Identify installed source even when the package is outside a Git checkout."""
    hashes = {}
    for root in (Path(__file__).parent, Path(marl_envs.__file__).parent):
        for path in sorted(root.rglob("*.py")):
            hashes[f"{root.name}/{path.relative_to(root)}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_checkpoint(path: Path, device: str = "cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    metadata = saved.get("metadata", {})
    if metadata.get("schema_version") != 1 or metadata.get("algorithm") not in RECIPES:
        raise ValueError("Checkpoint needs reconstruction metadata from the packaged TarMAC/IPPO trainer.")
    constructor = dict(metadata["constructor"])
    if metadata["algorithm"] == "tarmac":
        constructor["config"] = TarMACConfig(**constructor["config"])
        agent = TarMACAgent(**constructor).to(device)
        agent.load_state_dict(saved["agent"])
    else:
        agent = IPPOAgent(**constructor).to(device)
        agent.load_state_dict(saved["model"])
    agent.eval()
    return agent, metadata


def evaluate_checkpoint(path: Path, *, episodes: int = EVALUATION_EPISODES,
                        seed: int = EVALUATION_SEED, device: str = "cpu") -> dict:
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive.")
    agent, meta = load_checkpoint(path, device)
    return RECIPES[meta["algorithm"]]._evaluate(
        agent, meta["env"], meta["n_agents"], meta["horizon"], seed, episodes,
        torch.device(device),
    )


def rule_references(episodes: int, seed: int) -> dict:
    """A local rule and an explicitly privileged oracle, on the same target bits."""
    results = {}
    for rule in ("local_rule", "perfect_information"):
        env = make_env("target_signaling", 3, 1, seed)
        returns, successes = [], []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            # Only the leader reads its bit; followers always guess zero locally.
            actions = np.zeros(3, dtype=np.int64)
            actions[0] = int(obs[0, 3] > obs[0, 2])
            if rule == "perfect_information":
                actions[:] = actions[0]  # Extra information, not a decentralized policy.
            _, reward, _, _, info = env.step(actions)
            returns.append(float(reward))
            successes.append(float(info["success"]))
        env.close()
        results[rule] = {"returns": returns, "successes": successes}
    return results


def train_run(algorithm: str, out: Path, *, episodes: int = LEARNING_EPISODES,
              seed: int = 101, evaluation_episodes: int = EVALUATION_EPISODES,
              device: str = "cpu") -> dict:
    if algorithm not in RECIPES or episodes < 1 or evaluation_episodes < 1:
        raise ValueError("Choose tarmac/ippo and positive episode counts.")
    if any((out / name).exists() for name in ("result.json", "checkpoint.pt", "training.log")):
        raise FileExistsError(f"Choose a new output directory; {out} already contains a run.")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pt"
    started_at, started = utc_now_iso(), time.perf_counter()
    trainer = RECIPES[algorithm].train
    kwargs = dict(env="target_signaling", n_agents=3, horizon=1, episodes=episodes,
                  seed=seed, evaluation_seed=EVALUATION_SEED,
                  evaluation_episodes=evaluation_episodes, device=device,
                  checkpoint=str(checkpoint))
    if algorithm == "tarmac":
        kwargs["config"] = TarMACConfig(communication_rounds=2)
    resolved = {name: parameter.default for name, parameter in inspect.signature(trainer).parameters.items()}
    resolved.update(kwargs)
    resolved = {key: asdict(value) if is_dataclass(value) else value for key, value in resolved.items()}
    with (out / "training.log").open("w") as log, contextlib.redirect_stdout(log):
        summary = trainer(**kwargs)
    # The legacy IPPO navigation threshold has no meaning for this bounded game.
    summary.pop("validation_criterion", None)
    evaluation = evaluate_checkpoint(checkpoint, episodes=evaluation_episodes, device=device)
    if evaluation != summary["final_evaluation"]:
        raise RuntimeError("Reloaded checkpoint does not reproduce the trainer's evaluation.")
    success = float(np.mean(evaluation["successes"]))
    payload = {
        "schema_version": 1, "algorithm": algorithm, "seed": seed,
        "task": {"name": "target_signaling", "n_agents": 3, "horizon": 1},
        "resolved_config": resolved, "source_hashes": source_hashes(),
        "torch_threads": torch.get_num_threads(),
        "seconds": time.perf_counter() - started,
        "evaluation_seeds": list(range(EVALUATION_SEED, EVALUATION_SEED + evaluation_episodes)),
        "evaluation": evaluation, "references": rule_references(evaluation_episodes, EVALUATION_SEED),
        "training": summary,
        "learning_gate": {"minimum_success_rate": 0.8, "passed": success >= 0.8}
        if algorithm == "tarmac" and episodes == LEARNING_EPISODES else None,
    }
    write_json_with_provenance(out / "result.json", payload,
                               outputs=[checkpoint, out / "training.log"], started_at=started_at)
    print(f"{algorithm} seed {seed}: success={success:.3f}, {payload['seconds']:.1f}s; {out}", flush=True)
    return payload


def compare(out: Path, device: str = "cpu") -> bool:
    """Six fixed runs; failures remain recorded and never trigger seed replacement."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Comparison output must be empty: {out}")
    paths, passed = [], True
    for algorithm in RECIPES:
        for seed in TRAINING_SEEDS:
            run_dir = out / f"{algorithm}_seed{seed}"
            result = train_run(algorithm, run_dir, seed=seed, device=device)
            paths.append(run_dir / "result.json")
            if result["learning_gate"] is not None:
                passed &= result["learning_gate"]["passed"]
    write_json_with_provenance(out / "comparison.json", {
        "schema_version": 1, "results": [str(path.relative_to(out)) for path in paths],
        "tarmac_acceptance_passed": passed,
        "scope": "Illustrative native-method comparison, not an algorithm ranking.",
    }, inputs=paths)
    return passed


def plot(results: Path, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Plotting requires the demo extra: pip install 'modmarl[demo]'") from exc
    manifest = results / "comparison.json"
    paths = ([results / name for name in json.loads(manifest.read_text())["results"]]
             if manifest.exists() else [results / "result.json"])
    runs = [json.loads(path.read_text()) for path in paths]
    if not runs:
        raise ValueError("No saved results found.")
    # Different seeds may be aggregated; different protocols must never be mixed.
    for run in runs:
        if (run["task"] != runs[0]["task"] or run["evaluation_seeds"] != runs[0]["evaluation_seeds"]
                or run["resolved_config"]["episodes"] != runs[0]["resolved_config"]["episodes"]):
            raise ValueError("Results mix tasks, evaluation seeds, or training budgets.")
    grouped = {}
    for run in runs:
        grouped.setdefault(run["algorithm"], []).append(float(np.mean(run["evaluation"]["successes"])))
    for name, evaluation in runs[0]["references"].items():
        grouped[name] = [float(np.mean(evaluation["successes"]))]
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(grouped)
    ax.bar(names, [np.mean(grouped[name]) for name in names], alpha=0.6)
    for index, name in enumerate(names):
        ax.scatter([index] * len(grouped[name]), grouped[name], color="black", zorder=3)
    ax.set(ylim=(0, 1.05), ylabel="Team success rate", title="One-shot signaling: dots are training seeds")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    write_json_with_provenance(out / "summary.json", {
        "success_by_training_seed": grouped,
        "scope": "Means and individual seeds; reference policies are not trained.",
    }, inputs=paths, outputs=[out / "comparison.png"])
    print(f"Wrote {out / 'comparison.png'} and {out / 'summary.json'}")


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train the signaling example (32 episodes for a smoke check)")
    train_parser.add_argument("--algorithm", choices=RECIPES, default="tarmac")
    train_parser.add_argument("--episodes", type=_positive, default=LEARNING_EPISODES)
    train_parser.add_argument("--seed", type=int, default=101)
    train_parser.add_argument("--evaluation-episodes", type=_positive, default=EVALUATION_EPISODES)
    comparison = commands.add_parser("compare", help="Run the frozen six-run comparison")
    evaluation = commands.add_parser("evaluate", help="Evaluate an existing checkpoint")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--episodes", type=_positive, default=EVALUATION_EPISODES)
    evaluation.add_argument("--seed", type=int, default=EVALUATION_SEED)
    plotting = commands.add_parser("plot", help="Regenerate a figure and summary from saved results")
    plotting.add_argument("--results", type=Path, required=True)
    for subparser in (train_parser, comparison, evaluation, plotting):
        subparser.add_argument("--out", type=Path, required=True)
    for subparser in (train_parser, comparison, evaluation):
        subparser.add_argument("--device", default="cpu")
        subparser.add_argument("--threads", type=_positive, default=1)
    args = parser.parse_args()
    if args.command != "plot":
        torch.set_num_threads(args.threads)
    try:
        if args.command == "train":
            train_run(args.algorithm, args.out, episodes=args.episodes, seed=args.seed,
                      evaluation_episodes=args.evaluation_episodes, device=args.device)
        elif args.command == "compare":
            if not compare(args.out, args.device):
                parser.exit(1, "Comparison recorded; TarMAC did not meet the frozen success criterion.\n")
        elif args.command == "evaluate":
            if args.out.exists():
                raise FileExistsError(f"Evaluation output already exists: {args.out}")
            result = evaluate_checkpoint(args.checkpoint, episodes=args.episodes, seed=args.seed, device=args.device)
            write_json_with_provenance(args.out, {"evaluation": result, "seed": args.seed,
                                       "episodes": args.episodes, "source_hashes": source_hashes()},
                                       inputs=[args.checkpoint])
            print(f"Success={np.mean(result['successes']):.3f}; {args.out}")
        else:
            plot(args.results, args.out)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
