"""Write result artifacts with reproducible run and lineage metadata.

Model: one primary JSON result owns metadata and environment sidecars plus hashes for
declared inputs and outputs. Invariants: collection failures for Git, hardware, or package
state degrade to recorded nulls rather than losing the result. Interface:
``write_json_with_provenance`` is the sole writing entry point; helpers normalize values,
capture the host, and link parent artifact records.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

UTC = timezone.utc  # noqa: UP017 -- standalone cluster collectors run Python 3.9

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_SCHEMA_VERSION = 2


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_check_output(command: Sequence[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def _safe_run(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any] | None:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _path_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record

    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    record.update(
        {
            "size_bytes": path.stat().st_size,
            "sha256": sha256.hexdigest(),
            "artifact_id": sha256.hexdigest(),
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
        }
    )
    return record


def _metadata_sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.metadata.json")


def _load_parent_record(path: Path) -> dict[str, Any] | None:
    metadata_path = _metadata_sidecar_path(path)
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return None
    run = metadata.get("run", {})
    primary_output = metadata.get("primary_output", {})
    return {
        "path": str(path),
        "metadata_path": str(metadata_path),
        "run_id": run.get("run_id"),
        "artifact_id": primary_output.get("artifact_id"),
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    # An installed wheel has no checkout. Do not attribute an enclosing, unrelated
    # repository to the package merely because Git searches parent directories.
    if not (repo_root / ".git").exists():
        return {"repo_root": None, "branch": None, "commit": None,
                "short_commit": None, "dirty": None, "status_short": []}
    status_output = _safe_check_output(["git", "status", "--short"], cwd=repo_root) or ""
    branch = _safe_check_output(["git", "branch", "--show-current"], cwd=repo_root)
    commit = _safe_check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
    short_commit = _safe_check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    return {
        "repo_root": str(repo_root),
        "branch": branch,
        "commit": commit,
        "short_commit": short_commit,
        "dirty": bool(status_output.strip()),
        "status_short": status_output.splitlines(),
    }


def _python_environment() -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version("modmarl")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "modmarl_version": package_version,
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def _hardware_environment() -> dict[str, Any]:
    cuda_device_names: list[str] = []
    if torch.cuda.is_available():
        cuda_device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "cwd": os.getcwd(),
        "cpu_info": {
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch_cuda": {
            "is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": cuda_device_names,
        },
        "nvidia_smi": _safe_run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def _derive_script_name(argv: Sequence[str]) -> str | None:
    argv_list = list(argv)
    if "-m" in argv_list:
        index = argv_list.index("-m")
        if index + 1 < len(argv_list):
            return argv_list[index + 1]
    if argv_list:
        return Path(argv_list[0]).name
    return None


def _run_id(
    *,
    output_path: Path,
    started_at: str | None,
    argv: Sequence[str],
    git_commit: str | None,
) -> str:
    identity = {
        "output_path": str(output_path.resolve()),
        "started_at": started_at,
        "argv": list(argv),
        "git_commit": git_commit,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def write_json_with_provenance(
    output_path: Path,
    payload: Mapping[str, Any],
    *,
    inputs: Sequence[str | Path] = (),
    outputs: Sequence[str | Path] = (),
    extra: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_compatible(payload), indent=2))

    metadata_path = output_path.with_name(f"{output_path.stem}.metadata.json")
    environment_path = output_path.with_name(f"{output_path.stem}.environment.txt")

    environment_snapshot = _safe_check_output([sys.executable, "-m", "pip", "freeze"])
    if environment_snapshot is None:
        # uv-created environments need not contain pip. Distribution metadata is
        # available in both editable environments and clean wheel installations.
        environment_snapshot = "\n".join(sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ))
    environment_path.write_text(environment_snapshot + ("\n" if environment_snapshot else ""))

    input_paths = [Path(path) for path in inputs]
    output_paths = [output_path, environment_path, *[Path(path) for path in outputs]]
    git_state = _git_state(REPO_ROOT)
    argv_list = list(argv or sys.argv)
    input_records = [_path_record(path) for path in input_paths]
    output_records = [_path_record(path) for path in output_paths]
    parent_records = [record for path in input_paths if (record := _load_parent_record(path)) is not None]
    run_id = _run_id(
        output_path=output_path,
        started_at=started_at,
        argv=argv_list,
        git_commit=git_state.get("commit"),
    )
    metadata = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "argv": argv_list,
        "run": {
            "run_id": run_id,
            "script": _derive_script_name(argv_list),
        },
        "git": git_state,
        "python": _python_environment(),
        "hardware": _hardware_environment(),
        "primary_output": output_records[0],
        "inputs": input_records,
        "outputs": output_records,
        "lineage": {
            "input_artifact_ids": [record.get("artifact_id") for record in input_records if record.get("artifact_id")],
            "output_artifact_ids": [record.get("artifact_id") for record in output_records if record.get("artifact_id")],
            "parent_run_ids": [record.get("run_id") for record in parent_records if record.get("run_id")],
            "parent_records": parent_records,
        },
        "extra": dict(extra or {}),
    }
    metadata_path.write_text(json.dumps(_json_compatible(metadata), indent=2))
    return metadata
