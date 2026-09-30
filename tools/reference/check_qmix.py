"""Compare five updates with the pinned PyMARL learner using identical weights.

The reference checkout is never edited. Its one in-place online availability mask
is replaced in memory by an equivalent out-of-place mask for modern autograd.
This checks a controlled update path, not stochastic training equivalence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F

from modmarl.algorithms.qmix import QMIXAgent, QMIXBatch
from modmarl.common.provenance import utc_now_iso, write_json_with_provenance

REVISION = "eaf6b063822e2bdf1992d72dc159a500f235dbc4"
ROOT = Path(__file__).resolve().parents[2]


class ReferenceBatch(dict):
    batch_size = 3
    max_seq_length = 5
    device = "cpu"


class Logger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, value, step):
        self.stats[key] = value


def _utilities(state):
    return {key.replace("gru.", "rnn."): value for key, value in state.items()}


def _mixer(state):
    aliases = {"hyper_w1": "hyper_w_1", "hyper_w2": "hyper_w_final",
               "hyper_b1": "hyper_b_1", "state_value": "V"}
    return {aliases[key.split(".")[0]] + "." + key.split(".", 1)[1]: value
            for key, value in state.items()}


def compare(reference: Path) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != REVISION:
        raise ValueError(f"expected reference {REVISION}, got {revision}")
    if subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("reference checkout has tracked modifications")
    sys.path.insert(0, str((reference / "src").resolve()))
    from controllers.basic_controller import BasicMAC

    source = (reference / "src/learners/q_learner.py").read_text()
    original = "            mac_out[avail_actions == 0] = -9999999"
    replacement = "            mac_out = mac_out.masked_fill(avail_actions == 0, -9999999)"
    if source.count(original) != 1:
        raise ValueError("reference compatibility patch does not match exactly once")
    module = ModuleType("pymarl_qlearner_compat")
    exec(compile(source.replace(original, replacement), module.__name__, "exec"), module.__dict__)
    args = SimpleNamespace(
        n_agents=2, n_actions=3, rnn_hidden_dim=8, state_shape=8, mixing_embed_dim=8,
        obs_last_action=True, obs_agent_id=True, agent="rnn", agent_output_type="q",
        action_selector="epsilon_greedy", epsilon_start=1., epsilon_finish=.05,
        epsilon_anneal_time=20000, mixer="qmix", lr=5e-4, optim_alpha=.99,
        optim_eps=1e-5, learner_log_interval=1, gamma=.99, double_q=True,
        grad_norm_clip=10., target_update_interval=200,
    )
    torch.manual_seed(71)
    ours = QMIXAgent(2, 4, 3, state_dim=8, hidden_dim=8, mixer_hidden_dim=8)
    logger = Logger()
    scheme = {"obs": {"vshape": 4}, "actions_onehot": {"vshape": (3,)}}
    ref = module.QLearner(BasicMAC(scheme, None, args), {}, logger, args)
    ref.mac.agent.load_state_dict(_utilities(ours.q_network.state_dict()))
    ref.target_mac.agent.load_state_dict(_utilities(ours.target_q_network.state_dict()))
    ref.mixer.load_state_dict(_mixer(ours.mixer.state_dict()))
    ref.target_mixer.load_state_dict(_mixer(ours.target_mixer.state_dict()))
    records = []
    for update in range(5):
        actions = torch.randint(3, (3, 5, 2, 1))
        obs, state = torch.randn(3, 5, 2, 4), torch.randn(3, 5, 8)
        avail = torch.ones(3, 5, 2, 3)
        avail[:, 1:, 0, 2] = 0
        done, filled = torch.zeros(3, 5, 1), torch.ones(3, 5, 1)
        done[0, 1], filled[0, 2:] = 1, 0
        avail[0, 3:], obs[0, 3:], state[0, 3:] = 0, 0, 0
        reward = torch.randn(3, 5, 1)
        batch = ReferenceBatch(
            obs=obs, state=state, actions=actions,
            actions_onehot=F.one_hot(actions.squeeze(-1), 3).float(),
            avail_actions=avail, reward=reward, terminated=done, filled=filled,
        )
        our_batch = QMIXBatch(
            obs, state, actions[:, :-1].squeeze(-1), avail, reward[:, :-1, 0],
            done[:, :-1, 0], filled[:, :-1, 0].clone(),
        )
        metrics = ours.update(our_batch)
        ref.train(batch, update + 1, update + 1)
        errors = {}
        for name, left, right in (
            ("utilities", _utilities(ours.q_network.state_dict()), ref.mac.agent.state_dict()),
            ("mixer", _mixer(ours.mixer.state_dict()), ref.mixer.state_dict()),
        ):
            errors[name] = max(float((value - right[key]).abs().max()) for key, value in left.items())
        errors["loss"] = abs(float(metrics["loss"]) - logger.stats["loss"])
        if any(not torch.isfinite(torch.tensor(value)) or value > 1e-6 for value in errors.values()):
            raise AssertionError(f"update {update} differs: {errors}")
        records.append(errors)
    return {"reference_revision": revision, "updates": records, "tolerance": 1e-6,
            "scope": "identical weights; terminal/padded batches, availability, double Q, RMSProp",
            "compatibility_patch": {"original": original.strip(), "replacement": replacement.strip()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    torch.set_num_threads(1)
    started = utc_now_iso()
    result = compare(args.reference)
    sources = [Path(__file__), ROOT / "modmarl/algorithms/qmix.py"]
    sources.extend(args.reference / name for name in (
        "src/learners/q_learner.py", "src/controllers/basic_controller.py",
        "src/modules/agents/rnn_agent.py", "src/modules/mixers/qmix.py",
    ))
    write_json_with_provenance(args.out, result, inputs=sources, started_at=started)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
