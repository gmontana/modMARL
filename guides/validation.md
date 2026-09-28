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
| [maic](../modmarl/algorithms/maic.py) | [tests](../tests/test_maic.py) | fail (0/3 seeds pass) | 3/3 | Not recorded |
| [mappo](../modmarl/algorithms/mappo.py) | [tests](../tests/test_mappo.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [marc](../modmarl/algorithms/marc.py) | [tests](../tests/test_marc.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [masia](../modmarl/algorithms/masia.py) | [tests](../tests/test_masia.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [mat](../modmarl/algorithms/mat.py) | [tests](../tests/test_mat.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [mdmaddpg](../modmarl/algorithms/mdmaddpg.py) | [tests](../tests/test_mdmaddpg.py) | pass (3/3 seeds pass) | 0/3 | Not recorded |
| [ndq](../modmarl/algorithms/ndq.py) | [tests](../tests/test_ndq.py) | pass (3/3 seeds pass) | 3/3 | Not recorded |
| [qmix](../modmarl/algorithms/qmix.py) | [tests](../tests/test_qmix.py) | missing criterion (0/3 seeds pass) | 0/3 | Not recorded |
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
- **cacom:** Seed 7 finishes below its random-policy reference. Learning acceptance is unresolved.
- **cdc:** Recorded return and ablation margins pass; task success remains zero under the saved success metric. Reference code is private; see the implementation reconciliation.
- **cmvc:** Paper-based implementation; the linked release was empty at reconciliation. Its recorded criterion only requires beating random, not reproducing the journal results.
- **commformer:** Follows the released proceedings configuration and documented quirks. Local navigation curves do not reproduce the published benchmark tables.
- **expocomm:** Three-agent navigation demonstrates bounded learning, not the paper's large-team scalability or performance tables. Deliberate paper/release differences are documented in the module.
- **ic3net:** The hidden-gate runs pass the return rule but record zero task success; some execute without communication. Seed 3 has no deterministic return improvement over initialization. Useful communication is not established by these curves.
- **intention_sharing:** Reconciled against private author code; public users should consult the module's explicit decisions.
- **ippo:** Third-party reference supplies implementation details; recorded navigation acceptance does not apply to the new bounded signaling example.
- **iql:** Saved curves lack acceptance criteria; no threshold is inferred from the current trainer.
- **iwol:** The curves cover Im-IWoL (zero execution messages), not Ex-IWoL or the published robotics tables. Implementation follows the pinned v3 specification; the January 2026 v4 revision is not automatically validated.
- **magic:** The hidden-gate runs pass the return rule but record zero task success; seeds 3 and 5 execute without communication. Useful communication is not established by these curves.
- **maic:** All three saved win rates (43.3%, 77.7%, 51.3%) miss the stated 80% threshold. Message ablation has no effect for seed 11; investigate before claiming robust communication gains.
- **masia:** Hallway success uses a win reward of 1, whereas NDQ uses 10. The task name alone does not establish comparable configuration; there is no recorded message ablation.
- **mdmaddpg:** Reconciled against private author code; public users should consult the module's explicit decisions.
- **ndq:** High Hallway success and a message-ablation effect show reliance of these trained policies. A local fixed-deadline policy can also solve this deterministic task without messages.
- **qmix:** Saved curves lack acceptance criteria; no threshold is inferred from the current trainer.
- **schednet:** Seed 3 finishes below its random-policy reference. Learning acceptance is unresolved.
- **vdn:** Saved curves lack acceptance criteria; no threshold is inferred from the current trainer. Uses a third-party reference.

The 93 historical curve artifacts do not uniformly identify the generating modMARL
commit or full runtime environment. Their `source_revision` identifies the upstream
reference, not this checkout. New demo runs carry resolved configurations, source
hashes, runtime provenance, and checkpoint hashes. Historical results are preserved.
