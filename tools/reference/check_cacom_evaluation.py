"""Recover missing task metrics by re-evaluating an existing CACOM checkpoint.

The saved initial/random/final returns must match exactly. This repairs measurement
only, preserves the original result and cannot be counted as another training seed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.train_cacom import _entity_schema, _evaluate
from marl_envs import make_env
from modmarl.algorithms.cacom import CACOMAgent
from modmarl.common.provenance import utc_now_iso, write_json_with_provenance
from tools.check_validation import evaluate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    start = utc_now_iso()
    payload = json.loads(args.result.read_text())
    cfg = payload["resolved_config"]
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    env = make_env(cfg["env"], cfg["n_agents"], cfg["horizon"], cfg["seed"])
    agent = CACOMAgent(
        env.n_agents, env.obs_dim, env.num_actions,
        input_dim=env.obs_dim + env.num_actions + env.n_agents,
        entity_schema=_entity_schema(cfg["env"], env.n_agents, env.num_actions),
        **{key: cfg[key] for key in ("hidden_dim", "encode_dim", "request_dim", "response_dim",
                                   "bits", "mixer_hidden_dim")},
    )
    for name, epsilon in (("initial_evaluation", 0.0), ("random_evaluation", 1.0),
                          ("final_evaluation", 0.0)):
        final = name == "final_evaluation"
        if final:
            agent.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True))
        evaluation = _evaluate(
            agent, env, cfg["evaluation_episodes"], cfg["seed"] + 100000,
            torch.device("cpu"), epsilon=epsilon,
            force_all_links=not final or payload["total_steps"] < cfg["gate_start_steps"],
        )
        if evaluation["returns"] != payload[name]["returns"]:
            raise AssertionError(f"{name} changed; measurement-only repair is not established")
        payload[name] = evaluation
    payload["forced_links_evaluation"] = _evaluate(
        agent, env, cfg["evaluation_episodes"], cfg["seed"] + 100000,
        torch.device("cpu"), epsilon=0.0, force_all_links=True,
    )
    payload["acceptance"] = evaluate_run(payload)
    payload["measurement_repair"] = "same checkpoint and exact same returns; added task metrics"
    write_json_with_provenance(args.out, payload, started_at=start,
                               inputs=[args.result, Path(cfg["checkpoint"]), Path(__file__),
                                       Path("examples/train_cacom.py")])
    print(payload["acceptance"])


if __name__ == "__main__":
    main()
