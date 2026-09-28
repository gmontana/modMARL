# Vendored multi-agent particle environment

This directory is a vendored copy of OpenAI's `multiagent-particle-envs`
(<https://github.com/openai/multiagent-particle-envs>, MIT License, Copyright (c) 2018
OpenAI; see `LICENSE` in this directory). The package is renamed `multiagentsha` so that
it can coexist with any user-installed `multiagent` package. modMARL keeps it so that
the `paper_*` environments reproduce the exact dynamics that the CDC, I2C, MADDPG and
MD-MADDPG papers were evaluated on.

## Modifications relative to upstream

| File | Change |
|---|---|
| `__init__.py` | Removed two `gym.envs.registration.register` calls that pointed at an unshipped module. |
| `environment.py` | `import gym` / `from gym import spaces` replaced by their `gymnasium` equivalents; unused `EnvSpec` and `pdb` imports dropped; the `MultiDiscrete` construction and its size lookup use gymnasium's `nvec` form. Dynamics, observations and rewards are untouched. |
| `rendering.py` | `gym.utils.reraise` and `gym.error.Error` replaced by plain `ImportError` / `ValueError`. Rendering is imported lazily and needs `pyglet` and `six`, which modMARL does not declare. |
| `core.py`, `policy.py` | Carry the fork's earlier local edits (vision radius, partial-observation helpers); no modMARL changes. |
| `scenarios/__init__.py` | The `imp`-based loader was removed; `marl_envs.particle` loads scenarios by file path. |
| `scenarios/` | Only the five scenarios used by modMARL are kept. |

The gymnasium port was verified by replaying fixed-seed random-action trajectories for
every `paper_*` environment before and after the change and comparing observations, rewards
and per-agent rewards for exact equality.

## Scenario provenance

| Scenario | Origin |
|---|---|
| `simple_spread.py` | Upstream OpenAI scenario, adapted only to the `multiagentsha` namespace. |
| `simple_formation_po.py`, `simple_line_po.py`, `simple_spread_pack_leader_4_2.py` | Written by the authors of CDC and MD-MADDPG (Pesce & Montana) for their private fork of the OpenAI code; released here under this repository's MIT License. |
| `i2c_cooperative_navigation.py` | The authors' `cn.py` from <https://github.com/PKU-AI-Edge/I2C>, adapted only to the vendored namespace. |
