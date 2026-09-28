# AGENTS.md

modMARL — modular multi-agent RL library of cooperative MARL algorithms focused
on agent-to-agent communication, plus centralized-critic and independent
baselines. Python (>=3.11), setuptools package with `modmarl/` (algorithms,
components, common) and `marl_envs/` (environments) shipped together. Entry
points are the self-contained training examples `examples/train_<algo>.py`
(e.g. `python examples/train_cdc.py --episodes 400`).

## Build & Run

- Install: `pip install -e .` (extras: `.[dev]` pytest, `.[macpp]` gym,
  `.[pettingzoo]`).
- Lint: `ruff check .` (ruff is in the `dev` extra; configuration and the
  rule set live in `pyproject.toml`, which excludes `marl_envs/vendors`).
- Curve/regression evidence pipeline: `python tools/train_curves.py --out
  figures/curve_data --jobs 8 --clear` (GPU/multi-core; do not run casually).
  Every plotted panel is rebuilt from committed per-seed JSON.

## Test

- `pytest` — fast suite by default (pyproject `addopts = "-m 'not slow'"`).
- Slow seeded learning regressions: `pytest -o addopts='' -m slow`
  (currently `tests/test_learning.py`). CI runs lint, the fast suite, evidence
  checks, tutorials and installed-wheel smoke checks on Python 3.11 and 3.12.

## Conventions

- One module (or package) per algorithm under `modmarl/algorithms/`; generic
  non-algorithm communication mechanisms live in `modmarl/components/`;
  shared primitives (replay, metrics, on-policy helpers, provenance) in
  `modmarl/common/`.
- Each example defines an importable `train(**kwargs)` with keyword-only,
  fully defaulted hyperparameters and returns a metrics dict; the CLI parses
  only a demonstration subset. Module docstrings state which paper equations /
  sections the defaults reproduce.
- TarMAC and IPPO recipes live in `modmarl/training/` so the installed demo can
  use them; their `examples/train_*.py` modules remain compatible entry points.
- Check saved validation evidence with `python tools/check_validation.py --check`.
  Review any changed statuses before regenerating documentation with `--write`.
  The numerical criteria are scoped learning evidence, not blanket certification.
- Packaged demo: `python -m modmarl.demo train --episodes 32 --out runs/smoke`.
  The 32-episode recipe is an execution check; `compare` runs the frozen six-run
  recipe. Keep experiment records in `LABBOOK.md` and local outputs under `runs/`.
- When a paper's release contradicts its text, the discrepancy is documented in
  the module docstring and the release's quirks are reproduced, not tidied
  (see `modmarl/algorithms/commformer.py`).
- Algorithm-specific tasks/budgets/seeds are declared in `tools/train_curves.py`;
  behavioral tests check each algorithm against its paper semantics
  (`tests/test_<algo>.py`).

## Gotchas

- `marl_envs/vendors/` is third-party vendored code — excluded from ruff; do
  not reformat or refactor it. The only local edits are the gymnasium import
  substitutions listed in `marl_envs/vendors/multiagentsha/NOTICE.md`.
- `MACPPEnv` and `NoisyNavigationEnv` are constructed directly by their
  examples, not via `marl_envs.make_env`; MACPP needs the optional external
  package (`.[macpp]`).
- Post-publication CDC extensions (factorised/multiscale/spectral variants)
  were developed in a separate research repository and are not part of this
  repo's public API.
- Device defaults are CPU (`device: str = "cpu"` in examples); the curve
  pipeline is meant to run on a multi-core or GPU machine.

## Storage on the cluster

Before remote or GPU work, read the operator's private cluster notes for
current nodes, shared paths, environments and launch rules. Keep repositories,
datasets, model weights, results, archives and logs on shared storage. Use local
disks only for capped task scratch and remove it when finished. Set model caches
explicitly to shared storage before downloads and check free space first.
