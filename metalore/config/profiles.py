"""
Traffic profile definitions for the dynamic-traffic scenario.

IMPORTANT:
These values are scaled for your current main project, where typical job sizes are:
    UE data_size_mean ~= 100
    UE compute_size_mean ~= 10
    BS compute_capacity ~= 800 in high-density experiments

"""

from typing import Dict

from metalore.core.arrival.dynamic import TrafficProfile


TRAFFIC_PROFILES: Dict[str, TrafficProfile] = {
    "low": TrafficProfile(
        pool_size=15,
        job_data_size=60.0,
        job_compute_size=6.0,
        job_gen_prob=0.55,
        arrival_window=(0.0, 0.8),
        mean_sojourn=70.0,
    ),
    "medium": TrafficProfile(
        pool_size=25,
        job_data_size=100.0,
        job_compute_size=10.0,
        job_gen_prob=0.70,
        arrival_window=(0.0, 0.5),
        mean_sojourn=80.0,
    ),
    "high": TrafficProfile(
        pool_size=40,
        job_data_size=140.0,
        job_compute_size=14.0,
        job_gen_prob=0.85,
        arrival_window=(0.0, 0.2),
        mean_sojourn=90.0,
    ),
    "ramp_up": TrafficProfile(
        pool_size=30,
        job_data_size=120.0,
        job_compute_size=12.0,
        job_gen_prob=0.75,
        arrival_window=(0.2, 0.9),
        mean_sojourn=75.0,
    ),
    "ramp_down": TrafficProfile(
        pool_size=30,
        job_data_size=90.0,
        job_compute_size=9.0,
        job_gen_prob=0.65,
        arrival_window=(0.0, 0.2),
        mean_sojourn=35.0,
    ),
    "burst": TrafficProfile(
        pool_size=35,
        job_data_size=170.0,
        job_compute_size=17.0,
        job_gen_prob=0.85,
        arrival_window=(0.0, 0.9),
        mean_sojourn=45.0,
        arrival_type="gaussian",
        burst_fraction=0.5,
        burst_center=0.5,
        burst_std=0.08,
        burst_sojourn=30.0,
    ),
}
