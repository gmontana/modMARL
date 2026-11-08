# Changelog

All notable changes to modMARL are recorded here. The project follows
[Semantic Versioning](https://semver.org/): a new algorithm or public class is a
minor release, a fix that keeps the public API unchanged is a patch release.
The version is defined once in `modmarl/_version.py`; `CITATION.cff` repeats it
and the test suite checks that the two agree.

## [Unreleased]

### Added
- DIAL and RIAL (Foerster et al., NeurIPS 2016) as `DIALAgent`/`DIALConfig` in
  `modmarl/algorithms/dial.py`, reconciled against the authors' Apache-2.0 Lua
  release, with mechanism tests and the training example `examples/train_dial.py`.
  Three-seed curves and a validation-inventory entry are still to come.
- The switch riddle (`marl_envs.switch.SwitchRiddleEnv`), ported from the same
  release, with per-step action availability, channel mask and sender queries. It is
  constructed directly rather than through `make_env`.
- `modmarl.__version__`, read from `modmarl/_version.py`, which is also the
  single version source for the package build.
- This changelog, with a test that keeps `CITATION.cff` and the changelog in step
  with the package version.

### Changed
- `tools/check_validation.py` ignores modules under `marl_envs`, `modmarl/common` and
  `modmarl/components` that a confirmation's provenance never recorded. Confirmations
  record every module in those packages, so such a module was added afterwards and
  cannot have run; every recorded module must still match its hash.
- The optional DDPG, MADDPG-M and Intention Sharing learning probes now run the
  environment, seed and network settings of their committed curves, with episode
  budgets long enough for learning to appear. The previous 500-episode probes
  stopped before the committed curves themselves show any improvement.

### Documentation
- The Zenodo DOI is listed in the README and the citation file.

## [0.1.1] - 2026-10-02

### Added
- Citation identifiers, README badges, a security policy, a code of conduct and
  issue and pull-request templates.

### Changed
- Provenance records omit infrastructure details; the maintainers' lab book stays
  outside the published repository.

### Fixed
- The continuous-integration smoke test finds the built wheel by pattern.

## [0.1.0] - 2026-10-01

First public release: 31 cooperative multi-agent algorithms, covering learned
communication, centralized-critic and independent learners, with paper-semantics
tests, committed three-seed curves and recorded validation evidence.

[Unreleased]: https://github.com/gmontana/modMARL/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/gmontana/modMARL/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gmontana/modMARL/releases/tag/v0.1.0
