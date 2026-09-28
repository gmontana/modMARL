from .core import LeaderFollowerTargetEnv, TargetSignalingEnv
from .factory import ENV_CHOICES as BASE_ENV_CHOICES
from .factory import PettingZooSimpleSpreadEnv
from .hallway import NDQHallwayEnv
from .macpp_adapter import MACPPEnv, macpp_available
from .mpe import (
    BridgeNavigationControlEnv,
    DynamicPackControlEnv,
    FormationControlEnv,
    HiddenGateBridgeControlEnv,
    HiddenGateBridgeV2ControlEnv,
    LineControlEnv,
    NavigationControlEnv,
    SimpleSpreadMPEEnv,
)
from .noisy_navigation import NoisyNavigationEnv, SignedNoisyNavigationEnv
from .particle import PAPER_PARTICLE_ENVS, PaperParticleEnv, paper_particle_env_available


def pettingzoo_simple_spread_available() -> bool:
    try:
        PettingZooSimpleSpreadEnv(n_agents=3, horizon=5, seed=0).close()
    except ImportError:
        return False
    return True


ENV_CHOICES = BASE_ENV_CHOICES + ("signed_noisy_navigation",) + tuple(PAPER_PARTICLE_ENVS.keys())


def make_env(
    env_name: str,
    n_agents: int,
    horizon: int,
    seed: int,
    *,
    arena_size: float = 1.0,
    bridge_observe_radius: float | None = None,
    bridge_corridor_half_height: float | None = None,
    bridge_wall_margin: float | None = None,
    bridge_relay_fraction: float | None = None,
    bridge_corridor_capacity: int | None = None,
):
    if env_name == "signed_noisy_navigation":
        return SignedNoisyNavigationEnv(n_agents=n_agents, horizon=horizon, seed=seed)
    if env_name in PAPER_PARTICLE_ENVS:
        return PaperParticleEnv(env_name=env_name, horizon=horizon, seed=seed)
    from .factory import make_env as make_legacy_env

    return make_legacy_env(
        env_name,
        n_agents,
        horizon,
        seed,
        arena_size=arena_size,
        bridge_observe_radius=bridge_observe_radius,
        bridge_corridor_half_height=bridge_corridor_half_height,
        bridge_wall_margin=bridge_wall_margin,
        bridge_relay_fraction=bridge_relay_fraction,
        bridge_corridor_capacity=bridge_corridor_capacity,
    )


__all__ = [
    "ENV_CHOICES",
    "BridgeNavigationControlEnv",
    "DynamicPackControlEnv",
    "FormationControlEnv",
    "HiddenGateBridgeControlEnv",
    "HiddenGateBridgeV2ControlEnv",
    "LeaderFollowerTargetEnv",
    "LineControlEnv",
    "MACPPEnv",
    "NDQHallwayEnv",
    "NavigationControlEnv",
    "NoisyNavigationEnv",
    "PaperParticleEnv",
    "PettingZooSimpleSpreadEnv",
    "SignedNoisyNavigationEnv",
    "SimpleSpreadMPEEnv",
    "TargetSignalingEnv",
    "macpp_available",
    "make_env",
    "paper_particle_env_available",
    "pettingzoo_simple_spread_available",
]
