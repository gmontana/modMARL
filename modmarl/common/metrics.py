"""Task-agnostic deterministic-evaluation metrics.

Model: per-episode environment information is normalized before optional metrics are
aggregated into ``EvalStats``. Invariants: unavailable task metrics remain ``None`` and
success is never inferred from return. Interface: ``episode_info_with_defaults`` and
``aggregate_eval_stats`` form the collection boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class EvalStats:
    avg_return: float
    avg_final_distance: float | None = None
    success_rate: float | None = None
    avg_collisions: float | None = None
    avg_time_to_solve: float | None = None
    avg_targets_caught: float | None = None

    def to_json_dict(self) -> dict[str, float | None]:
        return asdict(self)


def episode_info_with_defaults(info: dict[str, Any], *, horizon: int, steps_taken: int) -> dict[str, Any]:
    data = dict(info)
    data.setdefault("mean_distance", None)
    data.setdefault("collisions", 0.0)
    data.setdefault("targets_caught", 0.0)
    if "success" not in data:
        data["success"] = None
    if data.get("success") is True:
        data.setdefault("time_to_solve", steps_taken)
    else:
        data.setdefault("time_to_solve", horizon)
    return data


def success_count(info: dict[str, Any]) -> int:
    success = info.get("success")
    if success is None:
        return 0
    return int(bool(success))


def success_indicator(info: dict[str, Any]) -> int | str:
    success = info.get("success")
    if success is None:
        return "n/a"
    return int(bool(success))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def aggregate_eval_stats(episode_infos: list[dict[str, Any]], returns: list[float]) -> EvalStats:
    final_distances = [float(info["mean_distance"]) for info in episode_infos if info.get("mean_distance") is not None]
    successes = [float(bool(info["success"])) for info in episode_infos if info.get("success") is not None]
    collisions = [float(info["collisions"]) for info in episode_infos if info.get("collisions") is not None]
    times = [float(info["time_to_solve"]) for info in episode_infos if info.get("time_to_solve") is not None]
    targets = [float(info["targets_caught"]) for info in episode_infos if info.get("targets_caught") is not None]
    return EvalStats(
        avg_return=float(np.mean(returns)),
        avg_final_distance=_mean_or_none(final_distances),
        success_rate=_mean_or_none(successes),
        avg_collisions=_mean_or_none(collisions),
        avg_time_to_solve=_mean_or_none(times),
        avg_targets_caught=_mean_or_none(targets),
    )
