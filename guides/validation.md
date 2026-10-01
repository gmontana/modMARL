# Validation evidence

Generated from `validation/inventory.json` and committed artifacts with
`python tools/check_validation.py --write`. Check without training using `--check`.

**A numerical pass is scoped learning evidence, not correctness certification,**
**published-performance reproduction, or proof of communication benefit.**
The source links contain the paper/release specification and deliberate deviations.
Test links show mechanism checks; their existence does not assert a fresh test run.
Reference links identify recorded comparisons, not universal parity guarantees.

Relative-return rules mean `(final - random) / abs(random)`. Each rule must hold
on every listed training seed. Missing rules are not inferred from current code.
Unknown rules, malformed data and missing files are insufficient evidence.
Message ablation measures reliance of a trained policy; a separately trained
message-free policy is needed to assess attainable performance without messages.

The first table audits the original curve files. See [bounded learning checks](#bounded-learning-checks)
for subsequent confirmation, frozen recipes and the current learning status.

| Method / specification | Mechanism tests | Recorded numerical criteria | Ablation runs | Reference comparison |
|---|---|---|---|---|
| [atoc](../modmarl/algorithms/atoc/algorithm.py) | [tests](../tests/test_atoc.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [cacom](../modmarl/algorithms/cacom.py) | [tests](../tests/test_cacom.py) | fail (2/3 seeds pass) | 0/3 | Not recorded |
| [cdc](../modmarl/algorithms/cdc.py) | [tests](../tests/test_cdc.py) | pass (3/3 seeds pass) | 3/3 | Not recorded |
| [cmvc](../modmarl/algorithms/cmvc/algorithm.py) | [tests](../tests/test_cmvc.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [commformer](../modmarl/algorithms/commformer.py) | [tests](../tests/test_commformer.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [commnet](../modmarl/algorithms/commnet.py) | [tests](../tests/test_commnet.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [ddpg](../modmarl/algorithms/ddpg.py) | [tests](../tests/test_ddpg.py) | pass (3/3 seeds pass) | 0/3 | [artifact](../figures/reference_data/ddpg_parity_summary.json) |
| [expocomm](../modmarl/algorithms/expocomm.py) | [tests](../tests/test_expocomm.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [happo](../modmarl/algorithms/happo.py) | [tests](../tests/test_happo.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [i2c](../modmarl/algorithms/i2c.py) | [tests](../tests/test_i2c.py) | pass (3/3 seeds pass) | 3/3 | Not recorded |
| [ic3net](../modmarl/algorithms/ic3net.py) | [tests](../tests/test_ic3net.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [intention_sharing](../modmarl/algorithms/intention_sharing.py) | [tests](../tests/test_intention_sharing.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [ippo](../modmarl/algorithms/ippo.py) | [tests](../tests/test_ippo.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [iql](../modmarl/algorithms/iql.py) | [tests](../tests/test_iql.py) | missing criterion (0/3 seeds pass) | 0/3 | Not recorded |
| [iwol](../modmarl/algorithms/iwol.py) | [tests](../tests/test_iwol.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [maac](../modmarl/algorithms/maac.py) | [tests](../tests/test_maac.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [maddpg](../modmarl/algorithms/maddpg.py) | [tests](../tests/test_maddpg.py) | pass (3/3 seeds pass) | 0/3 | [artifact](../figures/reference_data/maddpg_openai_parity.json) |
| [maddpg_m](../modmarl/algorithms/maddpg_m.py) | [tests](../tests/test_maddpg_m.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [magic](../modmarl/algorithms/magic.py) | [tests](../tests/test_magic.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [maic](../modmarl/algorithms/maic.py) | [tests](../tests/test_maic.py) | fail (0/3 seeds pass) | 3/3 | [artifact](../figures/reference_data/maic_normalization/normalization.json) |
| [mappo](../modmarl/algorithms/mappo.py) | [tests](../tests/test_mappo.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [marc](../modmarl/algorithms/marc.py) | [tests](../tests/test_marc.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [masia](../modmarl/algorithms/masia.py) | [tests](../tests/test_masia.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [mat](../modmarl/algorithms/mat.py) | [tests](../tests/test_mat.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [mdmaddpg](../modmarl/algorithms/mdmaddpg.py) | [tests](../tests/test_mdmaddpg.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [ndq](../modmarl/algorithms/ndq.py) | [tests](../tests/test_ndq.py) | pass (3/3 seeds pass) | 3/3 | Not recorded |
| [qmix](../modmarl/algorithms/qmix.py) | [tests](../tests/test_qmix.py) | missing criterion (0/3 seeds pass) | 0/3 | [artifact](../figures/reference_data/qmix_update_parity.json) |
| [schednet](../modmarl/algorithms/schednet.py) | [tests](../tests/test_schednet.py) | fail (2/3 seeds pass) | 0/3 | Not recorded |
| [sms](../modmarl/algorithms/sms.py) | [tests](../tests/test_sms.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [tarmac](../modmarl/algorithms/tarmac.py) | [tests](../tests/test_tarmac.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [vdn](../modmarl/algorithms/vdn.py) | [tests](../tests/test_vdn.py) | missing criterion (0/3 seeds pass) | 0/3 | Not recorded |

## Failed recorded criteria

- **cacom, seed 7:** `return_improvement_fraction_of_abs_random` observed -0.6747; required 0.2.
- **maic, seed 11:** `minimum_win_rate` observed 0.4333; required 0.8.
- **maic, seed 13:** `minimum_win_rate` observed 0.7767; required 0.8.
- **maic, seed 17:** `minimum_win_rate` observed 0.5133; required 0.8.
- **schednet, seed 3:** `return_improvement_fraction_of_abs_random` observed -0.0490; required 0.2.

## Limits and unresolved cases

- **atoc:** Paper-based implementation with explicit underspecified architecture choices; no author release used for parity.
- **cacom:** Historical seed 7 fails. The optional paper-action gate mode addresses a demonstrated label discrepancy; its first fresh confirmation still fails one seed. Release labels remain the default. Current learning acceptance is unresolved.
- **cdc:** Recorded return and message-ablation margins pass, and every seed improves over initialization and random actions. Binary task success is zero; the scoped learning claim is continuous navigation reward/distance improvement, not task completion. Reference code is private; see the implementation reconciliation.
- **cmvc:** Paper-based implementation; the linked release was empty at reconciliation. Its recorded criterion only requires beating random, not reproducing the journal results.
- **commformer:** Follows the released proceedings configuration and documented quirks. Local navigation curves do not reproduce the published benchmark tables.
- **expocomm:** Three-agent navigation demonstrates bounded learning, not the paper's large-team scalability or performance tables. Deliberate paper/release differences are documented in the module.
- **ic3net:** The hidden-gate runs pass the return rule but record zero task success; some execute without communication. Seed 3 has no deterministic return improvement over initialization. Useful communication is not established by these curves. Fresh three-seed delayed-signaling confirmation is reported separately below; the hidden-gate task is not solved by these results.
- **intention_sharing:** Reconciled against private author code; public users should consult the module's explicit decisions.
- **ippo:** Third-party reference supplies implementation details; recorded navigation acceptance does not apply to the new bounded signaling example.
- **iql:** Saved curves lack acceptance criteria; no threshold is inferred from the current trainer. Fresh confirmation under explicit navigation criteria is reported below.
- **iwol:** The curves cover Im-IWoL (zero execution messages), not Ex-IWoL or the published robotics tables. Implementation follows the pinned v3 specification; the January 2026 v4 revision is not automatically validated.
- **magic:** The hidden-gate runs pass the return rule but record zero task success; seeds 3 and 5 execute without communication. Useful communication is not established by these curves. Fresh three-seed delayed-signaling confirmation is reported separately below; the hidden-gate task is not solved by these results.
- **maic:** Historical runs miss the 80% win threshold and predate the evaluation-normalization/replay-tail fixes. Repaired seed-11 development reaches 300/300 wins, also with messages disabled; fresh confirmation is pending. The official evaluation uses batch statistics across agents, so matching it does not establish strictly local message generation. The pinned encoder comparison is component-only, not whole-policy parity.
- **masia:** Hallway success uses a win reward of 1, whereas NDQ uses 10. The task name alone does not establish comparable configuration; there is no recorded message ablation.
- **mdmaddpg:** Reconciled against private author code; public users should consult the module's explicit decisions.
- **ndq:** High Hallway success and a message-ablation effect show reliance of these trained policies. A local fixed-deadline policy can also solve this deterministic task without messages.
- **qmix:** Historical curves have no saved criterion. Fresh navigation confirmation passes with gamma 0.9; gamma 0.99 failed a seed even at 30k episodes. Controlled updates match pinned PyMARL exactly with the documented modern-autograd compatibility patch.
- **schednet:** Fresh one-step confirmation passes all three seeds with the released replay-priority critic derivative and a 50k budget. Historical navigation and several shorter-budget panels fail; they remain preserved. This establishes task learning, not the paper benchmark results.
- **vdn:** Saved curves lack acceptance criteria; no threshold is inferred from the current trainer. Uses a third-party reference. Fresh confirmation under explicit navigation criteria is reported below.

The 93 historical curve artifacts do not uniformly identify the generating modMARL
commit or full runtime environment. Their `source_revision` identifies the upstream
reference, not this checkout. New demo runs carry resolved configurations, source
hashes, runtime provenance, and checkpoint hashes. Historical results are preserved.

## Bounded learning checks

Every pass below also requires improvement over the initialized policy.
Historical evidence is explicitly reused, not independent confirmation.
Fresh confirmation fixes the recipe and criteria before running three new seeds.
These checks establish learning on the listed task, not published benchmark
performance or communication benefit. Original failures above remain visible.

| Method | Learning check | Evidence basis | Frozen recipe |
|---|---|---|---|
| atoc | pass (3/3) | reused historical evidence | Historical curve data |
| cacom | fail (2/3) | fresh paper-label confirmation; seed 301 gate has not converged | [JSON](../validation/recipes/development/cacom-second-confirmation.json) |
| cdc | pass (3/3) | reused historical evidence | Historical curve data |
| cmvc | pass (3/3) | reused historical evidence | Historical curve data |
| commformer | pass (3/3) | reused historical evidence | Historical curve data |
| commnet | pass (3/3) | reused historical evidence | Historical curve data |
| ddpg | pass (3/3) | reused historical evidence | Historical curve data |
| expocomm | pass (3/3) | reused historical evidence | Historical curve data |
| happo | pass (3/3) | reused historical evidence | Historical curve data |
| i2c | pass (3/3) | reused historical evidence | Historical curve data |
| ic3net | pass (3/3) | fresh confirmation: delayed signaling, 20k episodes | [JSON](../validation/recipes/ic3net.json) |
| intention_sharing | pass (3/3) | reused historical evidence | Historical curve data |
| ippo | pass (3/3) | reused historical evidence | Historical curve data |
| iql | pass (3/3) | fresh confirmation: navigation, 5k episodes | [JSON](../validation/recipes/iql.json) |
| iwol | pass (3/3) | reused historical evidence | Historical curve data |
| maac | pass (3/3) | reused historical evidence | Historical curve data |
| maddpg | pass (3/3) | reused historical evidence | Historical curve data |
| maddpg_m | pass (3/3) | reused historical evidence | Historical curve data |
| magic | pass (3/3) | fresh confirmation: delayed signaling, 20k episodes | [JSON](../validation/recipes/magic.json) |
| maic | fail (0/3) | reused historical evidence | Historical curve data |
| mappo | pass (3/3) | reused historical evidence | Historical curve data |
| marc | pass (3/3) | reused historical evidence | Historical curve data |
| masia | pass (3/3) | reused historical evidence | Historical curve data |
| mat | pass (3/3) | reused historical evidence | Historical curve data |
| mdmaddpg | pass (3/3) | reused historical evidence | Historical curve data |
| ndq | pass (3/3) | reused historical evidence | Historical curve data |
| qmix | pass (3/3) | fresh navigation confirmation, 10k episodes with gamma 0.9 | [JSON](../validation/recipes/qmix.json) |
| schednet | pass (3/3) | fresh one-step confirmation; released scheduler derivative, 50k episodes | [JSON](../validation/recipes/schednet.json) |
| sms | pass (3/3) | reused historical evidence | Historical curve data |
| tarmac | pass (3/3) | reused historical evidence | Historical curve data |
| vdn | pass (3/3) | fresh confirmation: navigation, 20k episodes | [JSON](../validation/recipes/vdn.json) |

Reproduce new confirmation jobs from a checkout:

```bash
python -m tools.run_validation --protocol validation/recipes/ic3net.json \
  --seed 101 --out runs/ic3net-101
```

Each recipe declares its seeds, budget, task and numerical rules. Use a new
output directory per seed. Results include full resolved settings, source and
checkpoint hashes, environment versions, host and measured runtime.

These recipes use CPU training with one PyTorch thread per process. Seeds can
run concurrently in separate processes and output directories. Budgets vary
by method; inspect the recipe before launching. A GPU is not required.

A confirmation pass requires every registered seed to meet every criterion.
Inspect these results with `python tools/check_validation.py --learning --json`.
Use `--require-learning` to fail if any method lacks a passing learning check.
The checker also rejects fresh evidence whose recorded implementation hashes
differ from this checkout. Historical evidence has weaker provenance as noted above.

Binary signaling is a minimal communication learning check with two target states.
Its success rate does not measure generalization to new partners or large teams.
A pass on one task can coexist with failures elsewhere; preserved failed panels
and the experiment decisions are recorded in [LABBOOK.md](../LABBOOK.md).
