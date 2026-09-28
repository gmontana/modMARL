# Choose a starting point

The library includes recent research methods and established controls. Publication
year alone does not establish adoption or superiority. Choose by the communication
problem, implementation assumptions, and available evidence.

| Research question | Starting implementation | Local evidence and limits |
|---|---|---|
| Sparse communication as team size grows | [ExpoComm, ICLR 2025](../modmarl/algorithms/expocomm.py) | Three-seed, three-agent navigation evidence; not the paper's large-team scaling results |
| Learn coordination representations with or without messages at execution | [IWoL, 2025 preprint revised January 2026](../modmarl/algorithms/iwol.py) | Navigation evidence covers Im-IWoL; Ex-IWoL is implemented but not established by those curves |
| Learn a sparse attention graph with sequence-model policies | [CommFormer, ICLR 2024](../modmarl/algorithms/commformer.py) | Navigation evidence and mechanism tests; released architecture quirks remain explicit |
| Quantized requests and personalized responses | [CACOM, AAMAS 2024](../modmarl/algorithms/cacom.py) | Mechanism tests exist, but seed 7 fails the recorded learning criterion |
| Counterfactual selection of messages | [CMVC, journal 2025](../modmarl/algorithms/cmvc/algorithm.py) | Paper-based implementation; bounded evidence with a weak return-over-random criterion |
| Understand information transfer in a small example | [TarMAC](../modmarl/algorithms/tarmac.py) and [IPPO](../modmarl/algorithms/ippo.py) | Featured signaling tutorial; older methods deliberately chosen for explanation, not a current leaderboard |
| Establish no-message controls | [MAPPO](../modmarl/algorithms/mappo.py), [IPPO](../modmarl/algorithms/ippo.py), [QMIX](../modmarl/algorithms/qmix.py) | Match action spaces, recurrence, observation access and tuning before comparison |

Recent-source checks, 2026-09-27: [ExpoComm's author repository](https://github.com/LXXXXR/ExpoComm)
identifies its ICLR 2025 release. [IWoL's paper](https://arxiv.org/abs/2509.25550)
has a January 2026 v4 revision; this port follows the pinned v3 specification and
source revision documented in its module, not an implied audit of v4.
The catalogue is not an exhaustive list of 2026 methods or a claim of state of the art.

From an editable source checkout, try these short **execution checks**:

```bash
OMP_NUM_THREADS=1 python examples/train_expocomm.py --env navigation --episodes 8 --topology one_peer --checkpoint expocomm-smoke.pt
OMP_NUM_THREADS=1 python examples/train_iwol.py --env navigation --episodes 8 --mode implicit --checkpoint iwol-smoke.pt
```

Eight episodes are not learning evidence. The pinned, longer recipes live in
`tools/train_curves.py`. Read the [validation table](validation.md) before choosing
a method for a substantive comparison. For scalable messaging development, start
with the [ExpoComm modification tutorial](communication.md).
