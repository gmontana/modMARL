"""Acceptance rules are independent of training and cannot hide missing evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.check_validation import evaluate_run, inspect_evidence, render
from tools.train_curves import ALGORITHMS

ROOT = Path(__file__).resolve().parents[1]


def _payload():
    return {
        "episodes": 2, "returns": [0, 1], "source_revision": "test",
        "evaluation_seeds": [10, 11], "initial_evaluation": {"returns": [-2, -2]},
        "random_evaluation": {"returns": [-1, -1]},
        "final_evaluation": {"returns": [1, 1], "successes": [1, 1]},
        "validation_criterion": {"return_margin_over_random": 1.0},
    }


def test_pass_fail_and_missing_are_distinct():
    payload = _payload()
    assert evaluate_run(payload)["status"] == "pass"
    payload["validation_criterion"]["return_margin_over_random"] = 3
    assert evaluate_run(payload)["status"] == "fail"
    del payload["validation_criterion"]
    assert evaluate_run(payload)["status"] == "missing criterion"


@pytest.mark.parametrize("change", [
    {"final_evaluation": {"returns": [float("nan"), 0]}},
    {"evaluation_seeds": [10, 10]},
    {"episodes": 3},
    {"validation_criterion": {"unknown_rule": 1}},
    {"validation_criterion": [1]},
    {"validation_criterion": {"minimum_success_rate": 0.8}, "final_evaluation": {"returns": [1, 1]}},
])
def test_malformed_or_unknown_evidence_is_not_a_pass(change):
    payload = _payload()
    payload.update(change)
    assert evaluate_run(payload)["status"] == "insufficient evidence"


def test_relative_return_uses_random_magnitude_and_handles_zero():
    payload = _payload()
    payload["validation_criterion"] = {"return_improvement_fraction_of_abs_random": 0.2}
    assert evaluate_run(payload)["details"][0]["observed"] == 2
    payload["random_evaluation"]["returns"] = [0, 0]
    assert evaluate_run(payload)["status"] == "insufficient evidence"


@pytest.mark.parametrize("criterion,evaluation,field,values", [
    ("minimum_success_rate", "final_evaluation", "successes", [1]),
    ("minimum_success_rate", "final_evaluation", "successes", [-1, 3]),
    ("maximum_mean_distance", "final_evaluation", "mean_distances", [0]),
    ("return_margin_over_no_message", "message_ablated_evaluation", "returns", [-1]),
])
def test_task_and_ablation_metrics_cover_all_evaluation_seeds(criterion, evaluation, field, values):
    payload = _payload()
    payload["validation_criterion"] = {criterion: 0.8}
    payload.setdefault(evaluation, {})[field] = values
    assert evaluate_run(payload)["status"] == "insufficient evidence"


def test_inventory_and_document_match_recorded_evidence():
    inventory = json.loads((ROOT / "validation/inventory.json").read_text())
    assert set(inventory["algorithms"]) == {name for name, _ in ALGORITHMS}
    rows = inspect_evidence(ROOT, inventory)
    for row in rows:
        assert row["status"] == inventory["algorithms"][row["algorithm"]]["acknowledged_status"]
        assert row["status"] != "insufficient evidence"
    learning_rows = inspect_evidence(ROOT, inventory, learning=True)
    assert (ROOT / "guides/validation.md").read_text() == render(inventory, rows, learning_rows)


def test_learning_cannot_pass_an_unchanged_policy(tmp_path):
    payload = _payload()
    payload.update(algorithm="example", seed=3)
    payload["initial_evaluation"] = copy.deepcopy(payload["final_evaluation"])
    (tmp_path / "run.json").write_text(json.dumps(payload))
    (tmp_path / "source.py").touch()
    inventory = {"algorithms": {"example": {
        "source_revision": "test", "artifacts": "run.json", "seeds": [3],
        "implementation": "source.py", "tests": "source.py", "reference_comparisons": [],
    }}}
    assert inspect_evidence(tmp_path, inventory)[0]["status"] == "pass"
    assert inspect_evidence(tmp_path, inventory, learning=True)[0]["status"] == "fail"
    del payload["validation_criterion"]
    (tmp_path / "run.json").write_text(json.dumps(payload))
    assert inspect_evidence(tmp_path, inventory, learning=True)[0]["status"] == "missing criterion"


def test_missing_artifact_cannot_be_acknowledged_as_a_numerical_failure(tmp_path):
    inventory = json.loads((ROOT / "validation/inventory.json").read_text())
    inventory = copy.deepcopy(inventory)
    inventory["algorithms"] = {"maic": inventory["algorithms"]["maic"]}
    assert inspect_evidence(tmp_path, inventory)[0]["status"] == "insufficient evidence"


@pytest.mark.parametrize("mutation", ["result", "protocol", "dirty", "missing_metadata", "source"])
def test_confirmation_requires_matching_protocol_and_provenance(tmp_path, mutation):
    import hashlib

    protocol = {"algorithm": "example", "seeds": [3, 5, 7], "config": {"episodes": 2},
                "criteria": {"return_margin_over_random": 1.0}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "source.py").touch()
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/train_example.py").touch()
    sources = [{"path": str(tmp_path / name), "sha256": hashlib.sha256(b"").hexdigest()}
               for name in ("source.py", "examples/train_example.py")]
    inventory = {"algorithms": {"example": {
        "source_revision": "test", "artifacts": "run*.json", "seeds": [3, 5, 7],
        "implementation": "source.py", "tests": "source.py", "reference_comparisons": [],
        "learning": {"artifacts": "seed*/result.json", "protocol": "protocol.json"},
    }}}
    for seed in protocol["seeds"]:
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        payload = {**_payload(), "algorithm": "example", "seed": seed, "status": "completed",
                   "protocol": protocol, "resolved_config": {"episodes": 2}}
        path = directory / "result.json"
        path.write_text(json.dumps(payload))
        metadata = {"primary_output": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                    "git": {"commit": "abc", "dirty": False, "repo_root": str(tmp_path)},
                    "inputs": sources}
        (directory / "result.metadata.json").write_text(json.dumps(metadata))
    assert inspect_evidence(tmp_path, inventory, learning=True)[0]["status"] == "pass"
    if mutation == "result":
        with (tmp_path / "seed3/result.json").open("a") as handle:
            handle.write("\n")
    elif mutation == "protocol":
        protocol["criteria"]["return_margin_over_random"] = 0
        (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    elif mutation == "dirty":
        path = tmp_path / "seed3/result.metadata.json"
        metadata = json.loads(path.read_text())
        metadata["git"]["dirty"] = True
        path.write_text(json.dumps(metadata))
    elif mutation == "missing_metadata":
        (tmp_path / "seed3/result.metadata.json").unlink()
    else:
        (tmp_path / "source.py").write_text("# changed\n")
    assert inspect_evidence(tmp_path, inventory, learning=True)[0]["status"] == "insufficient evidence"
