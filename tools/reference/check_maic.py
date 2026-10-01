"""Check MAIC normalization against its pinned release and run a bounded pilot.

Run from an editable checkout with ``python -m tools.reference.check_maic``.
The reference is loaded from an explicitly supplied author checkout; it is not
vendored. This compares latent-encoder normalization only, not whole-policy parity:
the existing documented sender, entropy and action-supervision differences remain.
The optional pilot evaluates one checkpoint under both normalization conventions.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.train_maic import _evaluate, _evaluation_mode, train
from modmarl.algorithms.maic import MAICAgent
from modmarl.common.provenance import utc_now_iso, write_json_with_provenance

REVISION = "2bd47d105ccd64bfba1f1d71981f7723c59ac07f"
ROOT = Path(__file__).resolve().parents[2]


def compare_normalization(reference: Path) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != REVISION:
        raise ValueError(f"expected author revision {REVISION}, got {revision}")
    if subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("the author checkout must have no tracked modifications")
    path = reference / "src/modules/agents/maic_agent.py"
    spec = importlib.util.spec_from_file_location("official_maic_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(n_agents=3, n_actions=3, latent_dim=8,
                           rnn_hidden_dim=64, nn_hidden_size=64, attention_dim=32,
                           var_floor=0.002)
    torch.manual_seed(11)
    ours = MAICAgent(3, 1, 3).double()
    ours.network.embed_net.layers[1].running_mean.fill_(7)
    official = module.MAICAgent(7, args).double()
    aliases = {"gru.": "rnn.", "q_head.": "fc2.", "embed_net.layers.": "embed_net.",
               "inference_net.layers.": "inference_net.", "msg_net.layers.": "msg_net."}
    state = {}
    for key, value in ours.network.state_dict().items():
        mapped = key
        for old, new in aliases.items():
            if mapped.startswith(old):
                mapped = new + mapped[len(old):]
                break
        state[mapped] = value
    official.load_state_dict(state)
    observations = torch.tensor([[[1.], [4.], [9.]]], dtype=torch.float64)
    previous = observations.new_zeros(1, 3, 3)
    hidden = observations.new_zeros(1, 3, 64)
    inputs = torch.cat((observations, previous, torch.eye(3).double().unsqueeze(0)), -1)
    captured = []
    hook = official.embed_net.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().clone()),
    )
    with torch.no_grad():
        try:
            official(inputs.reshape(3, 7), hidden, 1, test_mode=True)
        finally:
            hook.remove()
        expected_mu = captured[0][:, :24].reshape(1, 3, 3, 8)
        differences = {}
        for mode in ("batch", "running"):
            with _evaluation_mode(ours, mode):
                actual = ours.step(observations, previous, hidden, deterministic=True)
            differences[mode] = float((actual.mu - expected_mu).abs().max())
    if differences["batch"] > 1e-10 or differences["running"] <= 1e-4:
        raise AssertionError(f"normalization comparison failed: {differences}")
    return {"reference_revision": revision, "max_absolute_latent_mean_error": differences,
            "scope": "same weights and inputs; encoder normalization only",
            "reference_test_uses_batch_statistics": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=0,
                        help="optional development pilot; zero runs only the source comparison")
    args = parser.parse_args()
    if args.episodes < 0:
        parser.error("episodes must be nonnegative")
    if args.out.exists():
        parser.error("choose a new output directory to preserve prior results")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    started_at = utc_now_iso()
    sources = [Path(__file__), ROOT / "examples/train_maic.py"]
    for package in ("modmarl", "marl_envs"):
        sources.extend(sorted((ROOT / package).rglob("*.py")))
    sources.extend(args.reference / name for name in (
        "src/modules/agents/maic_agent.py", "src/controllers/maic_controller.py",
        "src/run.py", "src/config/algs/maic.yaml", "src/config/envs/join1.yaml",
    ))
    reference_result = compare_normalization(args.reference)
    write_json_with_provenance(args.out / "normalization.json", reference_result,
                               inputs=sources, started_at=started_at)
    print(reference_result, flush=True)
    if not args.episodes:
        return
    kwargs = {name: parameter.default for name, parameter in inspect.signature(train).parameters.items()}
    checkpoint = args.out / "checkpoint.pt"
    kwargs.update(episodes=args.episodes, seed=11, checkpoint=str(checkpoint))
    wall_start = time.monotonic()
    summary = train(**kwargs)
    training_seconds = time.monotonic() - wall_start
    agent = MAICAgent(3, 1, 3)
    agent.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    evaluations = {}
    for mode in ("batch", "running"):
        evaluations[mode] = _evaluate(agent, "maic_hallway", 3, 20, 100011, 300,
                                      torch.device("cpu"), normalization=mode)
    if evaluations["batch"] != summary["final_evaluation"]:
        raise AssertionError("checkpoint reload must reproduce final evaluation")
    payload = {"status": "completed", "scope": "development pilot, not confirmation",
               "config": kwargs, "training_seconds": training_seconds,
               "normalization_comparison": reference_result, "training": summary,
               "checkpoint_evaluations": evaluations}
    write_json_with_provenance(args.out / "pilot.json", payload, inputs=sources,
                               outputs=[checkpoint], started_at=started_at)
    print({"training_seconds": training_seconds,
           "win_rates": {mode: result["win_rate"] for mode, result in evaluations.items()}}, flush=True)


if __name__ == "__main__":
    main()
