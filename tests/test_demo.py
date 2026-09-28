"""Installed-workflow behavior and the signaling information boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from marl_envs import make_env
from modmarl.algorithms.ippo import IPPOAgent
from modmarl.common.provenance import _git_state
from modmarl.demo import evaluate_checkpoint, load_checkpoint, plot, rule_references, train_run


@pytest.mark.parametrize("algorithm", ["tarmac", "ippo"])
def test_train_reload_and_plot_from_artifacts(tmp_path, algorithm):
    out = tmp_path / algorithm
    result = train_run(algorithm, out, episodes=16, evaluation_episodes=8)
    reloaded = evaluate_checkpoint(out / "checkpoint.pt", episodes=8)
    assert reloaded == result["evaluation"] == result["training"]["final_evaluation"]
    assert json.loads((out / "result.json").read_text())["source_hashes"]
    metadata = json.loads((out / "result.metadata.json").read_text())
    assert any(entry["path"].endswith("checkpoint.pt") and entry["sha256"] for entry in metadata["outputs"])
    assert result["learning_gate"] is None  # A smoke run never claims learning acceptance.
    plot(out, tmp_path / "figures")
    assert (tmp_path / "figures/comparison.png").is_file()
    with pytest.raises(FileExistsError):
        train_run(algorithm, out, episodes=16)


def test_evaluation_resets_tarmac_state_per_episode(tmp_path):
    out = tmp_path / "tarmac"
    train_run("tarmac", out, episodes=16, evaluation_episodes=2)
    joint = evaluate_checkpoint(out / "checkpoint.pt", episodes=4, seed=90)
    singles = [evaluate_checkpoint(out / "checkpoint.pt", episodes=1, seed=90 + i) for i in range(4)]
    assert joint["returns"] == [single["returns"][0] for single in singles]


def test_ippo_followers_cannot_observe_the_leaders_bit():
    env = make_env("target_signaling", 3, 1, 0)
    obs, _ = env.reset(seed=0)
    changed = obs.copy()
    changed[0, 2:4] = obs[0, 2:4][::-1]
    agent = IPPOAgent(5, 2, hidden_dims=(16, 8))
    # Check policy probabilities as well as actions: an argmax alone could hide leakage.
    with torch.no_grad():
        first = agent.actor(torch.from_numpy(obs))
        second = agent.actor(torch.from_numpy(changed))
    torch.testing.assert_close(first[1:], second[1:], rtol=0, atol=0)


def test_reference_policies_have_the_documented_information():
    references = rule_references(100, 42)
    env = make_env("target_signaling", 3, 1, 42)
    zeros = sum(env.reset(seed=42 + i)[0][0, 2] == 1 for i in range(100))
    assert np.mean(references["local_rule"]["successes"]) == zeros / 100
    assert set(references["perfect_information"]["successes"]) == {1.0}


def test_old_checkpoint_gets_actionable_metadata_error(tmp_path):
    checkpoint = tmp_path / "old.pt"
    torch.save({"model": {}}, checkpoint)
    with pytest.raises(ValueError, match="reconstruction metadata"):
        load_checkpoint(checkpoint)


def test_installed_provenance_does_not_discover_an_enclosing_repo(tmp_path):
    (tmp_path / ".git").mkdir()
    installed = tmp_path / "site-packages"
    installed.mkdir()
    assert _git_state(installed)["commit"] is None


def test_plot_rejects_incomparable_protocols(tmp_path):
    first = train_run("tarmac", tmp_path / "a", episodes=16, evaluation_episodes=2)
    first["task"]["horizon"] = 2
    (tmp_path / "b").mkdir()
    (tmp_path / "b/result.json").write_text(json.dumps(first))
    (tmp_path / "comparison.json").write_text(json.dumps({"results": ["a/result.json", "b/result.json"]}))
    with pytest.raises(ValueError, match="mix tasks"):
        plot(tmp_path, tmp_path / "plot")
