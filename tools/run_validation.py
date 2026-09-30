"""Run one frozen learning-validation job with its protocol and provenance.

Example: python -m tools.run_validation --protocol validation/recipes/maic.json
    --seed 101 --out runs/confirmation/maic-101
Protocols declare the task, budget, seeds and acceptance rules before execution.
Existing output directories are never overwritten. Run independent jobs in separate
processes; each uses one PyTorch CPU thread unless its config selects a GPU.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import time
import traceback
from pathlib import Path

import torch

from modmarl.common.provenance import utc_now_iso, write_json_with_provenance
from tools.check_validation import evaluate_run
from tools.train_curves import SOURCE_REVISIONS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.seed not in protocol["seeds"]:
        parser.error("seed is not registered in the protocol")
    if args.out.exists():
        parser.error("choose a new output directory; previous runs are preserved")
    algorithm = protocol["algorithm"]
    trainer = importlib.import_module(f"examples.train_{algorithm}")
    kwargs = {name: parameter.default
              for name, parameter in inspect.signature(trainer.train).parameters.items()}
    kwargs.update(protocol["config"], seed=args.seed)
    args.out.mkdir(parents=True)
    kwargs["checkpoint"] = str(args.out / "checkpoint.pt")
    torch.set_num_threads(1)
    started_at = utc_now_iso()
    sources = [Path(__file__), args.protocol, Path(trainer.__file__)]
    for package in ("modmarl", "marl_envs"):
        sources.extend(sorted((ROOT / package).rglob("*.py")))
    run = {"algorithm": algorithm, "seed": args.seed, "protocol": protocol,
           "resolved_config": kwargs, "status": "running"}
    write_json_with_provenance(args.out / "started.json", run,
                               inputs=sources, started_at=started_at)
    start = time.monotonic()
    try:
        summary = trainer.train(**kwargs)
        # The external, preregistered task criteria govern this run. Preserve the
        # trainer's defaults separately; never alter historical evidence in place.
        payload = {**summary, **run, "status": "completed",
                   "source_revision": SOURCE_REVISIONS[algorithm],
                   "training_seconds": time.monotonic() - start,
                   "trainer_validation_criterion": summary.get("validation_criterion"),
                   "validation_criterion": protocol["criteria"]}
        payload["acceptance"] = evaluate_run(payload)
        write_json_with_provenance(args.out / "result.json", payload,
                                   inputs=sources, outputs=[Path(kwargs["checkpoint"])],
                                   started_at=started_at)
        print(json.dumps({"algorithm": algorithm, "seed": args.seed,
                          "training_seconds": payload["training_seconds"],
                          "acceptance": payload["acceptance"]}), flush=True)
    except Exception:
        write_json_with_provenance(
            args.out / "failure.json",
            {**run, "status": "failed", "training_seconds": time.monotonic() - start,
             "traceback": traceback.format_exc()}, inputs=sources, started_at=started_at,
        )
        raise


if __name__ == "__main__":
    main()
