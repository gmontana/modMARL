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


def test_inventory_and_document_match_recorded_evidence():
    inventory = json.loads((ROOT / "validation/inventory.json").read_text())
    assert set(inventory["algorithms"]) == {name for name, _ in ALGORITHMS}
    rows = inspect_evidence(ROOT, inventory)
    for row in rows:
        assert row["status"] == inventory["algorithms"][row["algorithm"]]["acknowledged_status"]
        assert row["status"] != "insufficient evidence"
    assert (ROOT / "guides/validation.md").read_text() == render(inventory, rows)


def test_missing_artifact_cannot_be_acknowledged_as_a_numerical_failure(tmp_path):
    inventory = json.loads((ROOT / "validation/inventory.json").read_text())
    inventory = copy.deepcopy(inventory)
    inventory["algorithms"] = {"maic": inventory["algorithms"]["maic"]}
    assert inspect_evidence(tmp_path, inventory)[0]["status"] == "insufficient evidence"
