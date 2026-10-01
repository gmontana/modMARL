# Contributing

Install Python 3.11 or 3.12 and an editable checkout:

```bash
python -m pip install -e ".[dev]"
ruff check .
OMP_NUM_THREADS=1 pytest
python tools/check_validation.py --check
```

Start with the [environment](guides/custom_environment.md) or
[communication](guides/communication.md) tutorial. Keep algorithm-specific updates
in their own module, and retain readable training examples. No framework subclass
or registration system is required. Avoid changes to vendored environments except
documented compatibility fixes.

For a new method, cite its primary paper and reference revision; explain deliberate
deviations and unspecified choices in the module docstring. Include behavioral tests
for its defining mechanism, an importable training example, and honest evidence
of what has been checked. A short smoke run checks execution, not learning.
Useful baselines and recent communication methods are welcome when they address
a concrete gap; publication year alone is not an acceptance criterion.

For research runs, record the question, frozen protocol, costs, results (including
failures), and next decision in your lab book, kept outside the repository. Keep large outputs out of
Git. Do not regenerate the full curve catalogue for a local change. For a changed
validation claim, inspect `python tools/check_validation.py --json`, update the
inventory deliberately, then regenerate the public table with `--write`.

For new learning evidence, copy a JSON recipe from `validation/recipes/` and fix
the task, budget, acceptance rules and three unused training seeds before running.
Require improvement over both the initialized policy and random actions, plus a
task outcome such as success or distance. Develop on separate seeds first. Run
each registered seed from a clean committed checkout:

```bash
python -m tools.run_validation --protocol validation/recipes/ic3net.json \
  --seed 101 --out runs/ic3net-101
```

Keep all outcomes, including failed panels. Commit the result JSON and its
provenance/environment sidecars, and add a `learning` entry in the inventory that
points to the complete seed panel and frozen protocol. List imports from other
algorithm modules in `source_dependencies`; common components and environments
are checked automatically. Use `python tools/check_validation.py --learning --json`
to inspect outcomes. `--require-learning` requires all methods to pass; `--check`
also checks that the documentation matches. Do not silently replace a failed
seed, lower its threshold, or call a reused development run fresh confirmation.
The optional slow tests use those same recipes and registered seeds. Select a
method with `pytest -o addopts='' -m slow tests/test_learning.py -k ic3net`;
running the entire slow suite retrains all registered panels. Legacy
training-return probes for other methods remain separate from acceptance evidence.

Package verification:

```bash
python -m pip install build
python -m build
```

Test the resulting wheel in a separate environment outside the checkout, following
the README. CI performs the same smoke and checkpoint checks. Preparing artifacts
does not publish them to a package index.
