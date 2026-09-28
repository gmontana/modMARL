from __future__ import annotations

import json
from pathlib import Path

from modmarl.common.provenance import write_json_with_provenance


def test_environment_snapshot_without_pip(tmp_path, monkeypatch):
    from modmarl.common import provenance

    monkeypatch.setattr(provenance, "_safe_check_output", lambda *args, **kwargs: None)
    write_json_with_provenance(tmp_path / "result.json", {"value": 1})
    snapshot = (tmp_path / "result.environment.txt").read_text()
    assert "torch==" in snapshot
    assert "numpy==" in snapshot


def test_write_json_with_provenance_creates_sidecars(tmp_path: Path) -> None:
    output_json = tmp_path / "summary.json"
    extra_output = tmp_path / "artifact.txt"
    extra_output.write_text("artifact\n")

    metadata = write_json_with_provenance(
        output_json,
        {"metric": 1.23},
        outputs=[extra_output],
        extra={"kind": "unit-test"},
        started_at="2026-03-31T00:00:00+00:00",
        argv=["python", "-m", "demo"],
    )

    metadata_path = tmp_path / "summary.metadata.json"
    environment_path = tmp_path / "summary.environment.txt"

    assert output_json.exists()
    assert metadata_path.exists()
    assert environment_path.exists()

    written = json.loads(output_json.read_text())
    metadata_json = json.loads(metadata_path.read_text())

    assert written["metric"] == 1.23
    assert metadata["extra"]["kind"] == "unit-test"
    assert metadata_json["started_at"] == "2026-03-31T00:00:00+00:00"
    assert metadata_json["argv"] == ["python", "-m", "demo"]
    assert metadata_json["schema_version"] == 2
    assert metadata_json["run"]["run_id"]
    assert metadata_json["primary_output"]["artifact_id"]
    assert any(record["path"] == str(output_json) for record in metadata_json["outputs"])
    assert any(record["path"] == str(extra_output) for record in metadata_json["outputs"])


def test_write_json_with_provenance_links_parent_runs(tmp_path: Path) -> None:
    parent_output = tmp_path / "parent.json"
    write_json_with_provenance(
        parent_output,
        {"value": 1},
        started_at="2026-03-31T00:00:00+00:00",
        argv=["python", "-m", "parent"],
    )

    child_output = tmp_path / "child.json"
    child_metadata = write_json_with_provenance(
        child_output,
        {"value": 2},
        inputs=[parent_output],
        started_at="2026-03-31T00:01:00+00:00",
        argv=["python", "-m", "child"],
    )

    assert child_metadata["lineage"]["parent_run_ids"]
    assert child_metadata["lineage"]["input_artifact_ids"]
    assert child_metadata["lineage"]["parent_records"][0]["path"] == str(parent_output)
