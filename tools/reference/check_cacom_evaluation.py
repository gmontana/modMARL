"""Recover missing task metrics by re-evaluating an existing CACOM checkpoint.

The saved initial/random/final returns must match exactly. This repairs measurement
only, preserves the original result and cannot be counted as another training seed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.train_cacom import _entity_schema, _evaluate, _gate_optimizer, _update_gate
from marl_envs import make_env
from modmarl.algorithms.cacom import CACOMAgent
from modmarl.common.provenance import utc_now_iso, write_json_with_provenance
from modmarl.common.replay import EpisodeReplayBuffer
from tools.check_validation import evaluate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-gate-updates", type=int, default=0,
                        help="Optional two-state diagnostic: fit only the gate with the policy frozen")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    if args.fit_gate_updates < 0:
        parser.error("gate update count must be nonnegative")
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
    if cfg["env"] == "target_signaling" and env.n_agents == 2:
        diagnostics = []
        with torch.no_grad():
            for target in (0, 1):
                env.reset(seed=0)
                env.target_bit = target
                obs = torch.from_numpy(env._observe()).unsqueeze(0)
                inputs = torch.cat([obs, torch.zeros(1, 2, 2), torch.eye(2).unsqueeze(0)], -1)
                hidden = agent.init_hidden(1, torch.device("cpu"))
                features, requests = agent.network.encode(inputs, hidden)
                on = torch.zeros(1, 2, 2, 1)
                on[:, 0, 1] = 1
                on_messages, logits, _ = agent.network.communication(features, requests, forced_mask=on)
                off_messages, _, _ = agent.network.communication(features, requests, forced_mask=on * 0)
                q_on, _ = agent.network.policy(features, on_messages, hidden)
                q_off, _ = agent.network.policy(features, off_messages, hidden)
                diagnostics.append({"target": target, "q_on": q_on.tolist(), "q_off": q_off.tolist(),
                                    "gate_probabilities": logits.softmax(-1).tolist(),
                                    "local_gap": float(q_on[:, 1].max() - q_off[:, 1].max())})
        payload["gate_diagnostics"] = diagnostics
    payload["acceptance"] = evaluate_run(payload)
    payload["measurement_repair"] = "same checkpoint and exact same returns; added task metrics"
    outputs = []
    if args.fit_gate_updates:
        if cfg["env"] != "target_signaling" or cfg["n_agents"] != 2:
            parser.error("the fixed-state gate diagnostic requires two-agent target signaling")
        checkpoint = args.out.with_suffix(".gate-fit.pt")
        if checkpoint.exists():
            parser.error("derived checkpoint already exists")
        # Both observed states were present in training. Labels come from the
        # current policy's counterfactual values, never from the target bit.
        replay = EpisodeReplayBuffer(2, 1, 2, env.obs_dim)
        for target in (0, 1):
            env.reset(seed=0)
            env.target_bit = target
            obs = env._observe()
            actions = np.zeros(2, dtype=np.int64)
            next_obs, reward, done, _, _ = env.step(actions)
            replay.add_episode(
                obs=np.stack([obs, next_obs]), actions=actions[None],
                rewards=np.asarray([reward]), dones=np.asarray([done], dtype=np.float32),
            )
        gate_ids = {id(parameter) for parameter in agent.gate_parameters()}
        fixed = [(parameter, parameter.detach().clone()) for parameter in agent.parameters()
                 if id(parameter) not in gate_ids]
        optimizer = _gate_optimizer(agent.gate_parameters(), cfg["gate_lr"])
        losses = []
        for update in range(args.fit_gate_updates):
            loss = _update_gate(
                agent, optimizer, replay.sample(cfg["batch_episodes"], torch.device("cpu")),
                helper=update % 2, mode=cfg.get("gate_label_mode", "release"),
            )
            if update % 100 == 0 or update == args.fit_gate_updates - 1:
                losses.append({"update": update + 1, "loss": loss})
        assert all(torch.equal(parameter, original) for parameter, original in fixed)
        evaluation = _evaluate(agent, env, cfg["evaluation_episodes"], cfg["seed"] + 100000,
                               torch.device("cpu"), epsilon=0.0, force_all_links=False)
        payload["gate_fit"] = {
            "scope": "derived checkpoint, fixed-policy calibration on reused states; not confirmation",
            "updates": args.fit_gate_updates, "fresh_optimizer": "released RMSProp",
            "policy_parameters_unchanged": True, "losses": losses,
            "evaluation": evaluation,
            "acceptance": evaluate_run({**payload, "final_evaluation": evaluation}),
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), checkpoint)
        outputs.append(checkpoint)
    write_json_with_provenance(args.out, payload, started_at=start,
                               inputs=[args.result, Path(cfg["checkpoint"]), Path(__file__),
                                       Path("examples/train_cacom.py")], outputs=outputs)
    print(payload["acceptance"])
    if "gate_fit" in payload:
        print(payload["gate_fit"]["acceptance"])


if __name__ == "__main__":
    main()
