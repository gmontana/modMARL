"""Modify ExpoComm's peer schedule through its explicit network interface.

python examples/communication_topology.py --topology ring
This untrained forward-pass example demonstrates wiring, not performance.
"""

from __future__ import annotations

import argparse

import torch

from modmarl.algorithms.expocomm import ExpoCommNetwork, exponential_peer_indices


def run(topology="ring"):
    torch.manual_seed(0)
    n_agents, obs_dim, hidden_dim = 5, 4, 16
    network = ExpoCommNetwork(obs_dim, 3, n_agents * obs_dim, hidden_dim, topology="one_peer")
    hidden = torch.zeros(1, n_agents, hidden_dim)
    messages = torch.zeros_like(hidden)
    with torch.no_grad():
        for timestep in range(4):
            obs = torch.randn(1, n_agents, obs_dim)
            # Receiver i reads sender peers[i]'s PREVIOUS message. The ring is
            # a tutorial modification, not the paper's exponential schedule.
            peers = ((torch.arange(n_agents) + 1) % n_agents if topology == "ring"
                     else exponential_peer_indices(n_agents, timestep))
            q_values, hidden, messages = network.step(obs, hidden, messages, peers)
            print(f"step={timestep}, receiver->sender={peers.tolist()}, actions={q_values.argmax(-1).tolist()}")
    return q_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("ring", "exponential"), default="ring")
    torch.set_num_threads(1)
    run(parser.parse_args().topology)
