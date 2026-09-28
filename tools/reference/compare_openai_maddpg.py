"""Compare modMARL MADDPG curves with the untouched OpenAI release.

OpenAI records the sum of every agent's identical collaborative reward, while
modMARL records that team reward once.  Divide the reference curve by the agent
count before comparing equal 100-episode windows.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def compare_curves(ours_returns: np.ndarray, official_returns: np.ndarray) -> dict:
    """Aggregate raw local episodes to the reference reporting window."""
    window = ours_returns.size // official_returns.size
    if window * official_returns.size != ours_returns.size:
        raise ValueError("modMARL episodes must divide evenly into OpenAI reporting windows")
    ours = ours_returns.reshape(official_returns.size, window).mean(axis=1)
    return {
        "window_episodes": window,
        "modmarl_team_returns": ours.tolist(),
        "openai_team_returns": official_returns.tolist(),
        "pearson_correlation": float(np.corrcoef(ours, official_returns)[0, 1]),
        "mean_absolute_error": float(np.abs(ours - official_returns).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, help="Template containing {seed} for modMARL JSON")
    parser.add_argument("--official", required=True, help="Template containing {seed} for OpenAI reward pickle")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 13, 17])
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    comparisons = []
    for seed in args.seeds:
        ours_payload = json.loads(Path(args.ours.format(seed=seed)).read_text())
        ours_returns = np.asarray(ours_payload["returns"], dtype=np.float64)
        with Path(args.official.format(seed=seed)).open("rb") as handle:
            official = np.asarray(pickle.load(handle), dtype=np.float64) / args.agents
        comparisons.append({"seed": seed, **compare_curves(ours_returns, official)})

    output = {
        "algorithm": "maddpg",
        "environment": "OpenAI MPE simple_spread",
        "official_repository": "https://github.com/openai/maddpg",
        "official_revision": "3ceefa0ada3ff31d633dd0bde8ff95213ce99be3",
        "normalization": "OpenAI reward sum divided by three identical collaborative agent rewards",
        "comparisons": comparisons,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
