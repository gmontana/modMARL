"""Check saved evidence without training or importing torch.

``--write`` regenerates the public table; ``--check`` verifies both that table
and the explicitly acknowledged statuses in the evidence inventory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mean(evaluation: dict, field: str) -> float:
    values = evaluation[field]
    if not isinstance(values, list) or not values:
        raise ValueError(f"Missing nonempty {field} list")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError(f"Non-finite or non-numeric {field}")
    return statistics.mean(values)


def evaluate_run(payload: dict) -> dict:
    """Interpret recorded rules only; never invent a threshold for old artifacts."""
    try:
        if not isinstance(payload, dict):
            raise ValueError("Run must be a JSON object")
        final = payload["final_evaluation"]
        final_return = _mean(final, "returns")
        initial = _mean(payload["initial_evaluation"], "returns")
        random = _mean(payload["random_evaluation"], "returns")
        _mean(payload, "returns")
        seeds = payload["evaluation_seeds"]
        if (not isinstance(seeds, list) or len(set(seeds)) != len(seeds)
                or any(len(payload[k]["returns"]) != len(seeds)
                       for k in ("initial_evaluation", "random_evaluation", "final_evaluation"))):
            raise ValueError("Evaluation seed/count mismatch")
        if payload["episodes"] != len(payload["returns"]) or not payload["source_revision"]:
            raise ValueError("Episode count or source revision missing/inconsistent")
        criteria = payload.get("validation_criterion")
        if not criteria:
            return {"status": "missing criterion", "details": []}
        if not isinstance(criteria, dict):
            raise ValueError("Criteria must be an object")
        checks = []
        for name, threshold in criteria.items():
            if name == "scope":
                continue
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                raise ValueError(f"Invalid threshold: {name}")
            if name in ("return_improvement_fraction_of_abs_random",
                        "minimum_improvement_fraction_of_absolute_random_return"):
                if random == 0:
                    raise ValueError("Relative improvement undefined for zero random return")
                actual = (final_return - random) / abs(random)
            elif name == "return_margin_over_random":
                actual = final_return - random
            elif name == "return_margin_over_initial":
                actual = final_return - initial
            elif name == "return_margin_over_no_message":
                actual = final_return - _mean(payload["message_ablated_evaluation"], "returns")
            elif name == "maximum_mean_distance":
                actual = _mean(final, "mean_distances")
            elif name in ("minimum_win_rate", "minimum_success_rate", "final_success_rate"):
                actual = _mean(final, "successes")
                if not 0 <= actual <= 1:
                    raise ValueError("Success fraction outside [0, 1]")
            elif name == "minimum_final_mean_return":
                actual = final_return
            else:
                raise ValueError(f"Unrecognized criterion: {name}")
            passed = actual <= threshold if name.startswith("maximum_") else actual >= threshold
            checks.append({"criterion": name, "observed": actual, "threshold": threshold, "passed": passed})
        if not checks:
            raise ValueError("No numerical acceptance rules")
        return {"status": "pass" if all(check["passed"] for check in checks) else "fail", "details": checks}
    except (KeyError, TypeError, ValueError, statistics.StatisticsError) as exc:
        return {"status": "insufficient evidence", "details": [str(exc)]}


def inspect_evidence(root: Path, inventory: dict) -> list[dict]:
    rows = []
    for algorithm, entry in sorted(inventory["algorithms"].items()):
        records, problems = [], []
        for path in sorted(root.glob(entry["artifacts"])):
            try:
                payload = json.loads(path.read_text())
                if payload["algorithm"] != algorithm or payload["source_revision"] != entry["source_revision"]:
                    raise ValueError("Algorithm/source revision mismatch")
                record = evaluate_run(payload)
                record.update(seed=payload["seed"], ablation="message_ablated_evaluation" in payload)
                records.append(record)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f"{path.name}: {exc}")
        if sorted(record["seed"] for record in records) != sorted(entry["seeds"]):
            problems.append("Training seeds do not match inventory")
        for path in [entry["implementation"], entry["tests"], *entry["reference_comparisons"]]:
            if not (root / path).is_file():
                problems.append(f"Missing evidence file: {path}")
        statuses = {record["status"] for record in records}
        if problems or "insufficient evidence" in statuses:
            status = "insufficient evidence"
        elif "fail" in statuses:
            status = "fail"
        elif "missing criterion" in statuses:
            status = "missing criterion"
        else:
            status = "pass"
        rows.append({"algorithm": algorithm, "status": status, "runs": records, "problems": problems})
    return rows


def render(inventory: dict, rows: list[dict]) -> str:
    lines = [
        "# Validation evidence", "",
        "Generated from `validation/inventory.json` and committed artifacts with",
        "`python tools/check_validation.py --write`. Check without training using `--check`.", "",
        "**A numerical pass is scoped learning evidence, not correctness certification,**",
        "**published-performance reproduction, or proof of communication benefit.**",
        "The source links contain the paper/release specification and deliberate deviations.",
        "Test links show mechanism checks; their existence does not assert a fresh test run.",
        "Reference links identify recorded comparisons, not universal parity guarantees.", "",
        "Relative-return rules mean `(final - random) / abs(random)`. Each rule must hold",
        "on every listed training seed. Missing rules are not inferred from current code.",
        "Unknown rules, malformed data and missing files are insufficient evidence.",
        "Message ablation measures reliance of a trained policy; a separately trained",
        "message-free policy is needed to assess attainable performance without messages.", "",
        "| Method / specification | Mechanism tests | Recorded numerical criteria | Ablation runs | Reference comparison |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        entry = inventory["algorithms"][row["algorithm"]]
        passing = sum(run["status"] == "pass" for run in row["runs"])
        ablations = sum(run["ablation"] for run in row["runs"])
        refs = ", ".join(f"[artifact](../{path})" for path in entry["reference_comparisons"]) or "Not recorded"
        lines.append(f"| [{row['algorithm']}](../{entry['implementation']}) | [tests](../{entry['tests']}) "
                     f"| {row['status']} ({passing}/{len(entry['seeds'])} seeds pass) "
                     f"| {ablations}/{len(entry['seeds'])} | {refs} |")
    lines.extend(["", "## Failed recorded criteria", ""])
    for row in rows:
        for run in row["runs"]:
            if run["status"] == "fail":
                for check in run["details"]:
                    if not check["passed"]:
                        lines.append(f"- **{row['algorithm']}, seed {run['seed']}:** "
                                     f"`{check['criterion']}` observed {check['observed']:.4f}; "
                                     f"required {check['threshold']:g}.")
    lines.extend(["", "## Limits and unresolved cases", ""])
    for name, entry in sorted(inventory["algorithms"].items()):
        if entry.get("notes"):
            lines.append(f"- **{name}:** {entry['notes']}")
    lines.extend(["", "The 93 historical curve artifacts do not uniformly identify the generating modMARL",
                  "commit or full runtime environment. Their `source_revision` identifies the upstream",
                  "reference, not this checkout. New demo runs carry resolved configurations, source",
                  "hashes, runtime provenance, and checkpoint hashes. Historical results are preserved.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print seedwise observations and thresholds")
    args = parser.parse_args()
    inventory = json.loads((ROOT / "validation/inventory.json").read_text())
    rows = inspect_evidence(ROOT, inventory)
    rendered = render(inventory, rows)
    document = ROOT / "guides/validation.md"
    if args.write:
        document.parent.mkdir(exist_ok=True)
        document.write_text(rendered)
    if args.check:
        mismatches = [row["algorithm"] for row in rows
                      if row["status"] != inventory["algorithms"][row["algorithm"]]["acknowledged_status"]
                      or row["status"] == "insufficient evidence"]
        if mismatches or not document.exists() or document.read_text() != rendered:
            parser.exit(1, f"Evidence status or documentation changed: {mismatches}; review before regenerating.\n")
        print(f"Checked {len(rows)} methods; documented failures remain visible.")
    elif args.json:
        print(json.dumps(rows, indent=2))
    elif not args.write:
        print(rendered)


if __name__ == "__main__":
    main()
