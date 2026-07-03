from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from metalore.scenarios import SingleCellEnv, MultiCellEnv
from metalore.config import (
    default_config,
    small_config,
    large_config,
    high_density_config,
    high_density_multicell_config,
    mobile_sensor_config,
    multi_cell_config,
    dynamic_training_config,
    low_traffic_config,
    medium_traffic_config,
    high_traffic_config,
    ramp_up_traffic_config,
    ramp_down_traffic_config,
    burst_traffic_config,
)


def build_config(scenario: str):
    """Build a config for old named scenarios or the new dynamic training scenario."""
    scenario = scenario.lower().strip()

    if scenario == "default":
        return default_config(), SingleCellEnv
    if scenario == "small":
        return small_config(), SingleCellEnv
    if scenario == "large":
        return large_config(), SingleCellEnv
    if scenario == "mobile_sensor":
        return mobile_sensor_config(), SingleCellEnv
    if scenario == "multi_cell":
        return multi_cell_config(), MultiCellEnv
    if scenario == "high_density":
        return high_density_config(), SingleCellEnv
    if scenario == "high_density_multicell":
        return high_density_multicell_config(), MultiCellEnv
    if scenario == "dynamic_training":
        return dynamic_training_config(), SingleCellEnv
    if scenario == "traffic_low":
        return low_traffic_config(), SingleCellEnv
    if scenario == "traffic_medium":
        return medium_traffic_config(), SingleCellEnv
    if scenario == "traffic_high":
        return high_traffic_config(), SingleCellEnv
    if scenario == "traffic_ramp_up":
        return ramp_up_traffic_config(), SingleCellEnv
    if scenario == "traffic_ramp_down":
        return ramp_down_traffic_config(), SingleCellEnv
    if scenario == "traffic_burst":
        return burst_traffic_config(), SingleCellEnv

    raise ValueError(
        f"Unknown scenario '{scenario}'. Use: default, small, large, mobile_sensor, "
        f"multi_cell, high_density, high_density_multicell, dynamic_training"
    )


def make_env(scenario: str = "small", render_mode=None, seed=None, reset_rng_episode: bool | None = None):
    config, env_cls = build_config(scenario)
    return make_env_from_config(config, env_cls=env_cls, render_mode=render_mode, seed=seed, reset_rng_episode=reset_rng_episode)


def make_env_from_config(config: dict, env_cls=SingleCellEnv, render_mode=None, seed=None, reset_rng_episode: bool | None = None):
    config = deepcopy(config)
    if seed is not None:
        config["environment"]["seed"] = seed
    if reset_rng_episode is not None:
        config["environment"]["reset_rng_episode"] = reset_rng_episode
    return env_cls(config=config, render_mode=render_mode)


def describe_env(env) -> dict:
    return {
        "num_bs": env.num_bs,
        "num_ues": env.num_ues,
        "num_sensors": env.num_sensors,
        "max_steps": env.EP_MAX_TIME,
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "reward_config": env.config.get("reward", {}),
    }
