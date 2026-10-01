"""Render the learning-curve grid from tools/train_curves.py results.

One small-multiple panel per algorithm: per-seed returns are smoothed with a
rolling mean, the line is the mean across seeds and the band is the seed
min-max range. Panels on the shared `navigation` task share a y-scale for
readability; budgets and configurations differ, so this is not a controlled
algorithm comparison. MADDPG-M runs its own noisy-navigation
setting and keeps its own scale (marked in its title). Panel subtitles wrap
within each small multiple. Emits a light and a dark variant for the README's
<picture> block.

Run:
    python tools/plot_curves.py --curves figures/curve_data --out figures --require-complete
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# README order: chronological within communication methods, sequence-model policies,
# centralized critics, then baselines.
PANEL_ORDER = [
    "commnet", "maddpg_m", "atoc", "ic3net", "schednet", "tarmac", "mdmaddpg", "ndq",
    "i2c", "magic", "intention_sharing", "maic", "sms", "masia", "cdc",
    "cacom", "cmvc", "expocomm", "iwol",
    "mat", "commformer",
    "maddpg", "maac", "marc",
    "ddpg", "iql", "vdn", "qmix", "ippo", "mappo", "happo",
]
TITLES = {
    "commnet": "CommNet", "atoc": "ATOC", "ic3net": "IC3Net", "tarmac": "TarMAC",
    "i2c": "I2C", "schednet": "SchedNet", "cacom": "CACOM", "intention_sharing": "Intention Sharing",
    "magic": "MAGIC", "cdc": "CDC", "cmvc": "CMVC", "mdmaddpg": "MD-MADDPG",
    "maddpg_m": "MADDPG-M", "marc": "MARC",
    "commformer": "CommFormer",
    "iwol": "Im-IWoL",
    "maac": "MAAC", "maddpg": "MADDPG",
    "qmix": "QMIX", "ndq": "NDQ", "maic": "MAIC", "masia": "MASIA", "sms": "SMS", "expocomm": "ExpoComm",
    "vdn": "VDN", "iql": "IQL", "mappo": "MAPPO", "mat": "MAT",
    "happo": "HAPPO", "ippo": "IPPO",
    "ddpg": "DDPG",
}
ENV_TITLES = {
    "hidden_gate_bridge": "Hidden-Gate Bridge",
    "hidden_gate_bridge_v2": "Hidden-Gate Bridge V2",
    "macpp": "Collaborative Pick-and-Place",
    "navigation": "Navigation",
    "noisy_navigation": "Noisy Navigation",
    "paper_navigation": "Paper Navigation Control",
    "paper_mdmaddpg_navigation": "MD-MADDPG Cooperative Navigation",
    "paper_maddpg_navigation": "Paper MADDPG Simple Spread",
    "target_signaling": "Target Signaling",
}

MODES = {
    "light": {"line": "#2a78d6", "surface": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#d0d7de"},
    "dark": {"line": "#3987e5", "surface": "#0d1117", "ink": "#e6edf3", "muted": "#9198a1", "grid": "#30363d"},
}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def _load(curves_dir: Path, window: int) -> dict[str, dict[str, np.ndarray]]:
    return _load_paths(sorted(curves_dir.glob("*.json")), window)


def _load_inventory(path: Path, window: int) -> dict:
    """Use exactly the current learning panel for each method; retain old files."""
    root = path.resolve().parents[1]
    inventory = json.loads(path.read_text())
    paths = []
    for algorithm, entry in sorted(inventory["algorithms"].items()):
        current = {**entry, **entry.get("learning", {})}
        selected = sorted(root.glob(current["artifacts"]))
        if not selected:
            raise ValueError(f"missing selected curve data for {algorithm}")
        paths.extend(selected)
    return _load_paths(paths, window)


def _load_paths(paths: list[Path], window: int) -> dict:
    runs: dict[str, list[np.ndarray]] = defaultdict(list)
    environments: dict[str, str] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if not payload.get("source_revision"):
            raise ValueError(f"{path.name} does not record source_revision")
        algorithm = payload["algorithm"]
        environment = payload["env"]
        previous = environments.setdefault(algorithm, environment)
        if previous != environment:
            raise ValueError(
                f"{algorithm} panel mixes environments {previous!r} and {environment!r}"
            )
        runs[algorithm].append(_smooth(np.asarray(payload["returns"], dtype=float), window))
    series = {}
    for algorithm, seeds in runs.items():
        horizon = min(len(s) for s in seeds)
        stacked = np.stack([s[:horizon] for s in seeds])
        series[algorithm] = {
            "mean": stacked.mean(axis=0),
            "low": stacked.min(axis=0),
            "high": stacked.max(axis=0),
            "n_seeds": stacked.shape[0],
            "env": environments[algorithm],
        }
    return series


def _panel_title(algorithm: str, environment: str) -> str:
    method = TITLES.get(algorithm, algorithm)
    task = ENV_TITLES.get(environment, environment.replace("_", " ").title())
    subtitle = "\n".join(textwrap.wrap(f"Environment: {task}", width=30))
    return f"{method}\n{subtitle}"


def _validate_complete(series: dict) -> None:
    """Require every README panel and the same multi-seed evidence for each."""
    missing = [algorithm for algorithm in PANEL_ORDER if algorithm not in series]
    if missing:
        raise ValueError(f"missing curve data for {missing}")
    seed_counts = {data["n_seeds"] for data in series.values()}
    if len(seed_counts) != 1 or next(iter(seed_counts)) < 2:
        raise ValueError("every panel must contain the same number of multiple seeds")


def _render(series: dict, mode: str, out_path: Path) -> None:
    colors = MODES[mode]
    panels = [a for a in PANEL_ORDER if a in series]
    n_cols, n_rows = 5, int(np.ceil(len(panels) / 5))

    # Only runs on the same environment share a y-range. Returns from different
    # reward functions are not numerically comparable even when both tasks say navigation.
    limits = {}
    for environment in {series[a]["env"] for a in panels}:
        members = [a for a in panels if series[a]["env"] == environment]
        lows = np.concatenate([series[a]["low"] for a in members])
        highs = np.concatenate([series[a]["high"] for a in members])
        y_min, y_max = np.percentile(lows, 1), np.percentile(highs, 99)
        pad = 0.05 * max(y_max - y_min, 1e-8)
        limits[environment] = (y_min - pad, y_max + pad)

    # No shared x: the recurrent on-policy methods train on a longer episode budget.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.9 * n_rows))
    fig.patch.set_facecolor(colors["surface"])
    for ax, algorithm in zip(axes.flat, panels):
        data = series[algorithm]
        x = np.arange(len(data["mean"]))
        ax.set_facecolor(colors["surface"])
        ax.fill_between(x, data["low"], data["high"], color=colors["line"], alpha=0.18, linewidth=0)
        ax.plot(x, data["mean"], color=colors["line"], linewidth=1.8)
        ax.set_title(_panel_title(algorithm, data["env"]), color=colors["ink"], fontsize=10)
        ax.set_ylim(*limits[data["env"]])
        ax.grid(color=colors["grid"], linewidth=0.5, alpha=0.6)
        ax.tick_params(colors=colors["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(colors["grid"])
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    n_seeds = next(iter(series.values()))["n_seeds"]
    fig.supxlabel("episode", color=colors["muted"], fontsize=10)
    fig.supylabel("episode return", color=colors["muted"], fontsize=10)
    seed_note = f"mean and seed range over {n_seeds} seeds" if n_seeds > 1 else "single seed"
    fig.suptitle(f"modMARL learning curves — {seed_note}", color=colors["ink"], fontsize=13)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(out_path, dpi=200, facecolor=colors["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the learning-curve grid.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--curves", default="curves")
    source.add_argument("--inventory", type=Path,
                        help="Use current panels from a project's validation/inventory.json")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = (_load_inventory(args.inventory, args.window) if args.inventory
              else _load(Path(args.curves), args.window))
    if args.require_complete:
        _validate_complete(series)
    missing = [a for a in PANEL_ORDER if a not in series]
    if missing:
        print(f"note: no results for {missing}")
    _render(series, "light", out_dir / "training_curves_light.png")
    _render(series, "dark", out_dir / "training_curves_dark.png")
    print(f"wrote {out_dir}/training_curves_light.png and _dark.png")


if __name__ == "__main__":
    main()
