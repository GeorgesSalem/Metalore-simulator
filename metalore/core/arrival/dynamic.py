"""
Dynamic Arrival Pattern for MetaLore.

This keeps a pool of possible UEs but only activates each UE between
its stime and extime. It does not change the observation/action space,
so old PPO models can still be evaluated on this scenario.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from metalore.core.arrival.base import Arrival


@dataclass
class TrafficProfile:
    """Parameters that define one dynamic traffic profile."""
    pool_size: int
    job_data_size: float
    job_compute_size: float
    job_gen_prob: float
    arrival_window: Tuple[float, float]
    mean_sojourn: float
    arrival_type: str = "uniform"       # "uniform" or "gaussian"
    burst_fraction: float = 0.0          # fraction of pool in the burst group
    burst_center: float = 0.5            # Gaussian center as fraction of episode length
    burst_std: float = 0.08              # Gaussian std as fraction of episode length
    burst_sojourn: float = 0.0           # mean sojourn for burst group


class DynamicArrival(Arrival):
    """Assign UE arrivals and departures according to one fixed profile."""

    def __init__(self, profiles: Dict[str, TrafficProfile], profile: str, **kwargs):
        super().__init__(**kwargs)
        if profile not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(f"Unknown traffic profile '{profile}'. Available: {available}")
        self.profiles = profiles
        self.current_profile = profile

    def arrival(self, entities: Dict) -> None:
        profile = self.profiles[self.current_profile]
        ues = list(entities.values())
        if profile.arrival_type == "gaussian":
            self._gaussian_arrival(ues, profile)
        else:
            self._uniform_arrival(ues, profile)

    def departure(self, entities: Dict) -> None:
        T = self.ep_max_time
        profile = self.profiles[self.current_profile]
        ues = list(entities.values())

        if profile.arrival_type == "gaussian":
            n_burst = max(1, int(len(ues) * profile.burst_fraction))
            for ue in ues[:n_burst]:
                sojourn = max(1, int(self.rng.exponential(profile.burst_sojourn)))
                ue.extime = min(T, ue.stime + sojourn)
            for ue in ues[n_burst:]:
                sojourn = max(1, int(self.rng.exponential(profile.mean_sojourn)))
                ue.extime = min(T, ue.stime + sojourn)
        else:
            for ue in ues:
                sojourn = max(1, int(self.rng.exponential(profile.mean_sojourn)))
                ue.extime = min(T, ue.stime + sojourn)

    @classmethod
    def with_profile(cls, profile: str, profiles: Dict[str, TrafficProfile]) -> type:
        """Return a DynamicArrival subclass with profile fixed in the constructor."""
        fixed_profiles = profiles

        class FixedDynamicArrival(cls):
            def __init__(self, **kwargs):
                super().__init__(profiles=fixed_profiles, profile=profile, **kwargs)

        FixedDynamicArrival.__name__ = f"DynamicArrival_{profile}"
        return FixedDynamicArrival

    def _uniform_arrival(self, ues: List, profile: TrafficProfile) -> None:
        T = self.ep_max_time
        w_start, w_end = profile.arrival_window
        for ue in ues:
            ue.stime = int(self.rng.uniform(T * w_start, T * w_end))

    def _gaussian_arrival(self, ues: List, profile: TrafficProfile) -> None:
        T = self.ep_max_time
        n_burst = max(1, int(len(ues) * profile.burst_fraction))
        center = int(T * profile.burst_center)
        std = T * profile.burst_std
        w_start, w_end = profile.arrival_window

        for ue in ues[:n_burst]:
            ue.stime = int(np.clip(self.rng.normal(center, std), 0, T - 1))
        for ue in ues[n_burst:]:
            ue.stime = int(self.rng.uniform(T * w_start, T * w_end))
