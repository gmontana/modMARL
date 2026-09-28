# Modify a communication mechanism

Use [ExpoComm (ICLR 2025)](https://github.com/LXXXXR/ExpoComm) as a recent starting
point. The port documents its paper/release differences in
[the algorithm module](../modmarl/algorithms/expocomm.py).

```bash
python examples/communication_topology.py --topology exponential
python examples/communication_topology.py --topology ring
```

The [runnable example](../examples/communication_topology.py) feeds explicit peer
indices to `ExpoCommNetwork.step`. Receiver `i` reads sender `peers[i]`'s previous
message. Replacing the rotating exponential schedule with a fixed ring changes
who receives which message while keeping the network weights and local update
structure identical. These are untrained forward passes, not learning results.

The network carries **both local hidden state and message memory**, each shaped
`(batch, agents, hidden_dim)`. Reset both at episode boundaries. Using another
agent's current hidden state directly would change the timing and information
available to the policy; it is not equivalent to changing peer indices.

For a trained variant, copy the complete ExpoComm recipe under a distinct experiment
name and modify the peer selection inside the agent's `step` method, which is used
for collection and online/target replay. Preserve target-network updates, message
grounding, replay masks, and agent/action features. The source recipe is
[here](../examples/train_expocomm.py); keep the original implementation available.
No new plugin system or common base class is necessary.

Before training, adapt the existing [topology and gradient tests](../tests/test_expocomm.py):
check the receiver/sender convention, blocked information paths, episode resets,
online/target consistency, and gradients into message parameters. Compare the new
variant with the unmodified method under the same task and budget. Label it as a
variant, not as the original paper implementation.
