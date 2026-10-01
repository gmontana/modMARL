"""Inspect message encoders and schedules on the two signaling states."""

import argparse
import json
from pathlib import Path

import torch

from marl_envs import make_env
from modmarl.algorithms.schednet import SchedNetAgent
from modmarl.common.provenance import utc_now_iso, write_json_with_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    start = utc_now_iso()
    cfg = json.loads(args.result.read_text())["resolved_config"]
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    env = make_env("target_signaling", cfg["n_agents"], 1, cfg["seed"])
    agent = SchedNetAgent(
        env.n_agents, env.obs_dim, env.num_actions,
        **{key: cfg[key] for key in ("message_dim", "actor_hidden_dim", "critic_hidden_dim",
                                   "scheduler_hidden_dim")}, bandwidth=1,
    )
    records = {}
    with torch.no_grad():
        for stage in ("initial", "trained"):
            if stage == "trained":
                agent.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location="cpu"))
            records[stage] = []
            for target in (0, 1):
                env.reset(seed=0)
                env.target_bit = target
                obs = torch.from_numpy(env._observe()).unsqueeze(0)
                action, _, weights, schedule = agent.act(obs, 1, deterministic=True)
                forced_action = agent.act(obs, 1, deterministic=True,
                                          priorities=torch.tensor([[1., 0.]]))[0]
                records[stage].append({
                    "target": target, "messages": agent.message_encoder(obs).tolist(),
                    "weights": weights.tolist(), "schedule": schedule.tolist(),
                    "actions": action.tolist(), "forced_leader_actions": forced_action.tolist(),
                })
    write_json_with_provenance(args.out, records, started_at=start,
                               inputs=[args.result, args.checkpoint, Path(__file__),
                                       Path("modmarl/algorithms/schednet.py")])
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
