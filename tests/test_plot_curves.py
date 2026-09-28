from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.plot_curves import PANEL_ORDER, _load, _panel_title, _validate_complete
from tools.train_curves import ALGORITHMS, SOURCE_REVISIONS, _clear_outputs, _selected_seeds

COMMUNICATION_METHODS = {
    "atoc", "cacom", "cdc", "cmvc", "commnet", "expocomm", "i2c", "ic3net", "intention_sharing",
    "iwol", "maddpg_m", "magic", "maic", "masia", "mdmaddpg", "ndq", "schednet",
    "sms", "tarmac",
}


def test_panel_title_always_names_the_environment() -> None:
    assert _panel_title("cdc", "paper_navigation") == (
        "CDC\nEnvironment: Paper Navigation\nControl"
    )
    assert _panel_title("qmix", "custom_task") == "QMIX\nEnvironment: Custom Task"


def test_panel_title_wraps_long_environment_names_within_a_panel() -> None:
    title = _panel_title("marc", "macpp")
    assert title == "MARC\nEnvironment: Collaborative\nPick-and-Place"
    assert all(len(line) <= 30 for line in title.splitlines()[1:])


def test_load_rejects_one_algorithm_mixed_across_environments(tmp_path) -> None:
    for seed, environment in enumerate(("navigation", "paper_navigation")):
        payload = {
            "algorithm": "cdc",
            "env": environment,
            "seed": seed,
            "source_revision": "paper:test",
            "returns": [-2.0, -1.0],
        }
        (tmp_path / f"cdc_{seed}.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="cdc panel mixes environments"):
        _load(tmp_path, window=1)


def test_load_rejects_curve_without_source_revision(tmp_path) -> None:
    payload = {"algorithm": "cdc", "env": "navigation", "returns": [-1.0]}
    (tmp_path / "cdc_seed3.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not record source_revision"):
        _load(tmp_path, window=1)


def test_explicit_curve_seeds_override_the_count() -> None:
    assert _selected_seeds(3, [3, 5, 7]) == [3, 5, 7]
    assert _selected_seeds(3, None) == [0, 1, 2]


def test_curve_registry_matches_every_trainable_panel_with_three_pinned_seeds() -> None:
    configs = dict(ALGORITHMS)
    assert set(configs) == set(PANEL_ORDER)
    assert all(len(config["_curve_seeds"]) == 3 for config in configs.values())
    assert set(SOURCE_REVISIONS) == set(PANEL_ORDER)
    assert all(SOURCE_REVISIONS.values())


def test_committed_communication_curves_record_execution_rate() -> None:
    curve_dir = Path(__file__).parents[1] / "figures" / "curve_data"
    seen = set()
    for path in curve_dir.glob("*.json"):
        payload = json.loads(path.read_text())
        algorithm = payload["algorithm"]
        if algorithm in COMMUNICATION_METHODS:
            assert "communication_rate" in payload, path.name
            seen.add(algorithm)
    assert seen == COMMUNICATION_METHODS


def test_complete_readme_figure_requires_every_panel_and_multiple_seeds() -> None:
    complete = {algorithm: {"n_seeds": 3} for algorithm in PANEL_ORDER}
    _validate_complete(complete)

    with pytest.raises(ValueError, match="missing curve data"):
        _validate_complete({"commnet": {"n_seeds": 3}})

    with pytest.raises(ValueError, match="same number of multiple seeds"):
        _validate_complete({algorithm: {"n_seeds": 1} for algorithm in PANEL_ORDER})


def test_clear_outputs_removes_only_selected_curve_artifacts(tmp_path) -> None:
    selected = tmp_path / "cdc_seed1.json"
    selected_error = tmp_path / "cdc_seed3.error"
    unrelated = tmp_path / "qmix_seed1.json"
    for path in (selected, selected_error, unrelated):
        path.write_text("result")

    _clear_outputs(tmp_path, ["cdc"])

    assert not selected.exists()
    assert not selected_error.exists()
    assert unrelated.exists()
