# Use your own environment

Start with the runnable [adapter example](../examples/custom_environment.py):

```bash
python examples/custom_environment.py
```

It adapts a dictionary-based simulator and collects 16 one-step episodes for one
TarMAC update. It checks integration, not convergence. No factory registration is
required: construct your environment directly and pass its dimensions to a learner.

The adapter exposes `n_agents`, `obs_dim`, `num_actions`, and `horizon`.
`reset(seed=...)` returns `(observations, info)`; `step(actions)` returns
`(observations, team_reward, terminated, truncated, info)`.
Observations are float32 arrays shaped `(agents, features)`, discrete actions have
shape `(agents,)`, and agent ordering stays stable across every array. The reward
is one team scalar for this cooperative learner. Other algorithms can require
per-agent rewards, continuous actions, action masks or additional training state;
consult their example instead of coercing them into this contract.

**Each observation row contains only information available to that actor.**
Global simulator state and other agents' private observations may inform a
centralized critic where the method allows it, but must not enter a local actor
through the adapter. Shared parameters do not grant shared observations.

The tutorial ends naturally after its decision (`terminated=True`). A rollout
cutoff is normally `truncated=True`; bootstrapping depends on the learner's stated
objective. For example, the existing IPPO recipe bootstraps time limits, whereas
TarMAC's paper recipe treats its finite horizon as terminal. Preserve that distinction
when adapting the complete [IPPO](../modmarl/training/ippo.py) or
[TarMAC](../modmarl/training/tarmac.py) loop. Reset all recurrent state at each episode.

Before collecting learning curves, check deterministic resets, dimensions, valid
actions, reward aggregation, and both boundary conditions. Run a scripted policy
to establish that success is possible. Keep evaluation seeds separate from training.
