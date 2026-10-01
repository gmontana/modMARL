"""Check saved evidence without training or importing torch.

``--write`` regenerates the public table; ``--check`` verifies both that table
and the explicitly acknowledged statuses in the evidence inventory.
"""

from __future__ import annotations

import argparse
import hashlib
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

        def evaluation_mean(evaluation: dict, field: str) -> float:
            value = _mean(evaluation, field)
            if len(evaluation[field]) != len(seeds):
                raise ValueError(f"Evaluation seed/count mismatch for {field}")
            return value

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
                actual = final_return - evaluation_mean(payload["message_ablated_evaluation"], "returns")
            elif name == "maximum_mean_distance":
                actual = evaluation_mean(final, "mean_distances")
            elif name in ("minimum_win_rate", "minimum_success_rate", "final_success_rate"):
                actual = evaluation_mean(final, "successes")
                if any(not 0 <= success <= 1 for success in final["successes"]):
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


def inspect_evidence(root: Path, inventory: dict, *, learning: bool = False) -> list[dict]:
    rows = []
    for algorithm, entry in sorted(inventory["algorithms"].items()):
        if learning:
            entry = {**entry, **entry.get("learning", {})}
        records, problems = [], []
        for path in sorted(root.glob(entry["artifacts"])):
            try:
                payload = json.loads(path.read_text())
                if payload["algorithm"] != algorithm or payload["source_revision"] != entry["source_revision"]:
                    raise ValueError("Algorithm/source revision mismatch")
                if learning:
                    if "protocol" in entry:
                        protocol = json.loads((root / entry["protocol"]).read_text())
                        if (payload.get("protocol") != protocol
                                or payload.get("status") != "completed"
                                or len(set(protocol["seeds"])) < 3
                                or sorted(entry["seeds"]) != sorted(protocol["seeds"])
                                or entry.get("criteria")
                                or payload["seed"] not in protocol["seeds"]
                                or payload.get("validation_criterion") != protocol["criteria"]
                                or any(payload["resolved_config"].get(key) != value
                                       for key, value in protocol["config"].items())):
                            raise ValueError("Result does not match frozen confirmation protocol")
                        metadata = json.loads(path.with_name(f"{path.stem}.metadata.json").read_text())
                        if metadata["primary_output"]["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                            raise ValueError("Result hash does not match provenance")
                        if not metadata["git"]["commit"] or metadata["git"]["dirty"]:
                            raise ValueError("Confirmation must identify a clean committed checkout")
                        source_root = Path(metadata["git"]["repo_root"])
                        recorded_sources = {}
                        for item in metadata["inputs"]:
                            source = Path(item["path"])
                            if source.is_absolute() and source.is_relative_to(source_root):
                                recorded_sources[str(source.relative_to(source_root))] = item["sha256"]
                        required = [root / entry["implementation"], root / f"examples/train_{algorithm}.py"]
                        required.extend(root / name for name in entry.get("source_dependencies", []))
                        for package in ("marl_envs", "modmarl/common", "modmarl/components"):
                            required.extend((root / package).rglob("*.py"))
                        for source in required:
                            name = str(source.relative_to(root))
                            if recorded_sources.get(name) != hashlib.sha256(source.read_bytes()).hexdigest():
                                raise ValueError(f"Confirmation source changed or is unrecorded: {name}")
                    criteria = dict(payload.get("validation_criterion") or {})
                    criteria.update(entry.get("criteria", {}))
                    # Beating random without improving the initialized policy is
                    # not a learning pass. Retain any stronger recorded margin.
                    if any(key != "scope" for key in criteria):
                        criteria["return_margin_over_initial"] = max(
                            1e-6, criteria.get("return_margin_over_initial", 0),
                        )
                    payload = {**payload, "validation_criterion": criteria}
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
        if learning:
            rows[-1]["basis"] = entry.get("basis", "reused historical evidence")
            rows[-1]["artifacts"] = entry["artifacts"]
    return rows


def render(inventory: dict, rows: list[dict], learning_rows: list[dict] | None = None) -> str:
    current_status = []
    if learning_rows is not None:
        passing_methods = sum(row["status"] == "pass" for row in learning_rows)
        current_status = [
            f"**Current bounded learning checks: {passing_methods}/{len(learning_rows)} methods pass.**",
            "See [current results and frozen recipes](#bounded-learning-checks);",
            "the first table below preserves the historical audit.", "",
        ]
    lines = [
        "# Validation evidence", "",
        "Generated from `validation/inventory.json` and committed artifacts with",
        "`python tools/check_validation.py --write`. Check without training using `--check`.", "",
        *current_status,
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
        "The first table audits the original curve files. See [bounded learning checks](#bounded-learning-checks)",
        "for subsequent confirmation, frozen recipes and the current learning status.", "",
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
    if learning_rows is not None:
        lines.extend([
            "## Bounded learning checks", "",
            "Every pass below also requires improvement over the initialized policy.",
            "Historical evidence is explicitly reused, not independent confirmation.",
            "Fresh confirmation fixes the recipe and criteria before running three new seeds.",
            "These checks establish learning on the listed task, not published benchmark",
            "performance or communication benefit. Original failures above remain visible.", "",
            "| Method | Learning check | Evidence basis | Frozen recipe |", "|---|---|---|---|",
        ])
        for row in learning_rows:
            passing = sum(run["status"] == "pass" for run in row["runs"])
            protocol = inventory["algorithms"][row["algorithm"]].get("learning", {}).get("protocol")
            recipe = f"[JSON](../{protocol})" if protocol else "Historical curve data"
            lines.append(f"| {row['algorithm']} | {row['status']} ({passing}/{len(row['runs'])}) "
                         f"| {row['basis']} | {recipe} |")
        lines.extend(["", "Reproduce new confirmation jobs from a checkout:", "", "```bash",
                      "python -m tools.run_validation --protocol validation/recipes/ic3net.json \\",
                      "  --seed 101 --out runs/ic3net-101", "```", "",
                      "Each recipe declares its seeds, budget, task and numerical rules. Use a new",
                      "output directory per seed. Results include full resolved settings, source and",
                      "checkpoint hashes, environment versions, host and measured runtime.", "",
                      "These recipes use CPU training with one PyTorch thread per process. Seeds can",
                      "run concurrently in separate processes and output directories. Budgets vary",
                      "by method; inspect the recipe before launching. A GPU is not required.", "",
                      "A confirmation pass requires every registered seed to meet every criterion.",
                      "Inspect these results with `python tools/check_validation.py --learning --json`.",
                      "Use `--require-learning` to fail if any method lacks a passing learning check.",
                      "The checker also rejects fresh evidence whose recorded implementation hashes",
                      "differ from this checkout. Historical evidence has weaker provenance as noted above.", "",
                      "Binary signaling is a minimal communication learning check with two target states.",
                      "Its success rate does not measure generalization to new partners or large teams.",
                      "A pass on one task can coexist with failures elsewhere; failed panels are",
                      "preserved alongside subsequent repairs.", "",
                      "Machine names, user names and storage paths were redacted from the recorded",
                      "provenance on 2026-10-01. Result files changed only in those strings, and their",
                      "recorded hashes were recomputed after the redaction; no metric was altered.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print seedwise observations and thresholds")
    parser.add_argument("--learning", action="store_true", help="Inspect bounded learning checks")
    parser.add_argument("--require-learning", action="store_true", help="Fail unless every learning check passes")
    args = parser.parse_args()
    inventory = json.loads((ROOT / "validation/inventory.json").read_text())
    rows = inspect_evidence(ROOT, inventory)
    learning_rows = inspect_evidence(ROOT, inventory, learning=True)
    rendered = render(inventory, rows, learning_rows)
    if args.require_learning:
        unresolved = [row["algorithm"] for row in learning_rows if row["status"] != "pass"]
        if unresolved:
            parser.exit(1, f"Unresolved learning checks: {unresolved}\n")
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
        print(json.dumps(learning_rows if args.learning else rows, indent=2))
    elif not args.write:
        print(rendered)


if __name__ == "__main__":
    main()
