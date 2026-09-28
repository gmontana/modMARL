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
failures), and next decision in the single `LABBOOK.md`. Keep large outputs out of
Git. Do not regenerate the full curve catalogue for a local change. For a changed
validation claim, inspect `python tools/check_validation.py --json`, update the
inventory deliberately, then regenerate the public table with `--write`.

Package verification:

```bash
python -m pip install build
python -m build
```

Test the resulting wheel in a separate environment outside the checkout, following
the README. CI performs the same smoke and checkpoint checks. Preparing artifacts
does not publish them to a package index.
