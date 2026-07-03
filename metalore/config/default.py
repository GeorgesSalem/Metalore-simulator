
import random
from metalore.core.arrival.dynamic import DynamicArrival
from metalore.config.profiles import TRAFFIC_PROFILES
from typing import Dict, Any, List
from copy import deepcopy

from metalore.core.movement.random_waypoint import RandomWaypointMovement
from metalore.core.movement.static import StaticMovement
from metalore.core.arrival.no_departures import NoDeparture
from metalore.core.channels.okumura_hata import OkumuraHata
from metalore.core.association.closest import ClosestAssociation
from metalore.core.schedulers.resource_fair import ResourceFair
from metalore.core.schedulers.round_robin import RoundRobin
from metalore.handlers.smart_city import SmartCityHandler
from metalore.utils.logger import SimulationLogger


DEFAULT_CONFIG: Dict[str, Any] = {

    "environment": {
        "width": 200.0,
        "height": 200.0,
        "max_steps": 100,
        "seed": 999,
        "reset_rng_episode": False,
        "num_ues": 3,
        "num_sensors": 3,
        "arrival_ue": NoDeparture,
        "arrival_sensor": NoDeparture,
        "movement_ue": RandomWaypointMovement,
        "movement_sensor": StaticMovement,
        "channel": OkumuraHata,
        "association": ClosestAssociation,
        "scheduler_ue": ResourceFair,
        "scheduler_sensor": ResourceFair,
        "handler": SmartCityHandler,
        "logger": SimulationLogger,
    },

    "bs": {
        "positions": [(100.0, 100.0)],
        "bandwidth": 100e6,
        "frequency": 3500,
        "tx_power": 40,
        "height": 40,
        "compute_capacity": 150,
    },

    "ue": {
        "velocity": 0.5,
        "height": 1.5,
        "snr_threshold": 2e-8,
        "noise": 1e-9,
    },

    "sensor": {
        "velocity": 0.0,
        "height": 1.5,
        "snr_threshold": 2e-8,
        "noise": 1e-9,
        "sensing_range": 40.0,
        "update_interval": 1,
    },

    "sensor_placement": {
        "min_distance": 20,
        "max_distance": 80,
        "margin": 10,
    },

    "job_ue": {
        "generation_probability": 0.7,
        "data_size_mean": 100.0,
        "compute_size_mean": 10.0,
    },

    "job_sensor": {
        "data_size_mean": 70.0,
        "compute_size_mean": 7.0,
    },

    "scheduler": {
        "type": "resource_fair",
        "quantum": 7e6,
    },

    "reward": {
        "delay_penalty": -2.0,
        "sync_base_reward": 10.0,
        "discount_factor": 0.70,
        "e2e_delay_threshold": 2.0,
    },

    "visualization": {
        "show_sensing_range": True,
        "show_connections": True,
        "show_labels": True,
        "figsize": (10, 10),
    },
}


def default_config() -> Dict[str, Any]:
    """Get a deep copy of the default configuration."""
    return deepcopy(DEFAULT_CONFIG)


def merge_config(base: Dict, override: Dict):
    """Merge two configuration dictionaries, with `override` taking precedence over `base`."""
    for key, value in override.items():
        if isinstance(value, dict):
            node = base.setdefault(key, {})
            merge_config(node, value)
        else:
            base[key] = value
    return base


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation."""
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def set_config_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set a configuration value using dot notation."""
    keys = path.split('.')
    target = config
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """Print configuration."""
    for key, value in config.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


# ---------------------------------------------------------------------------
# Original stable scenarios
# ---------------------------------------------------------------------------

def small_config() -> Dict[str, Any]:
    """Safe small-scale scenario."""
    config = default_config()
    config['environment']['num_ues'] = 5
    config['environment']['num_sensors'] = 5
    config['bs']['compute_capacity'] = 100
    return config


def large_config() -> Dict[str, Any]:
    """Configuration for larger single-BS dynamic scenario."""
    config = default_config()
    config['environment']['num_ues'] = 9
    config['environment']['num_sensors'] = 6
    config['bs']['compute_capacity'] = 150
    return config


def mobile_sensor_config() -> Dict[str, Any]:
    """Configuration with mobile sensors using random waypoint movement."""
    config = default_config()
    config['environment']['num_ues'] = 10
    config['environment']['num_sensors'] = 5
    config['environment']['movement_sensor'] = RandomWaypointMovement
    config['sensor']['velocity'] = 1.0
    return config


def multi_cell_config() -> Dict[str, Any]:
    """Configuration for multi-cell scenario."""
    config = default_config()
    config['bs']['positions'] = [
        (90.0, 50.0),
        (150.0, 120.0),
        (30.0, 120.0),
    ]
    config['environment']['num_ues'] = 15
    config['environment']['num_sensors'] = 20
    return config


def high_density_config() -> Dict[str, Any]:
    """Base high-density single-BS scenario."""
    config = default_config()
    config['environment']['num_ues'] = 40
    config['environment']['num_sensors'] = 20
    config['environment']['max_steps'] = 100
    config['bs']['positions'] = [(100.0, 100.0)]
    config['bs']['compute_capacity'] = 800
    config['bs']['bandwidth'] = 600e6
    return config


def high_density_multicell_config() -> Dict[str, Any]:
    """High-density multi-BS scenario."""
    config = default_config()
    config['bs']['positions'] = [
        (50.0, 50.0),
        (150.0, 50.0),
        (100.0, 150.0),
    ]
    config['environment']['num_ues'] = 50
    config['environment']['num_sensors'] = 30
    config['environment']['max_steps'] = 100
    config['bs']['compute_capacity'] = 300
    config['bs']['bandwidth'] = 100e6
    return config

# ---------------------------------------------------------------------------
# Dynamic traffic profiles: UE arrivals/departures over one episode
# ---------------------------------------------------------------------------

def dynamic_traffic_config(profile: str | None = None) -> Dict[str, Any]:
    """Single-BS dynamic-traffic scenario using UE arrivals/departures."""
    if profile is None:
        profile = random.choice(list(TRAFFIC_PROFILES.keys()))
    if profile not in TRAFFIC_PROFILES:
        available = ', '.join(sorted(TRAFFIC_PROFILES))
        raise ValueError(f"Unknown traffic profile '{profile}'. Available: {available}")

    p = TRAFFIC_PROFILES[profile]
    config = default_config()

    config['environment']['arrival_ue'] = DynamicArrival.with_profile(profile, profiles=TRAFFIC_PROFILES)
    config['environment']['arrival_sensor'] = NoDeparture
    config['environment']['num_ues'] = p.pool_size
    config['environment']['num_sensors'] = 25
    config['environment']['max_steps'] = 100

    config['bs']['positions'] = [(100.0, 100.0)]
    config['bs']['compute_capacity'] = 800
    config['bs']['bandwidth'] = 600e6

    config['job_ue']['generation_probability'] = p.job_gen_prob
    config['job_ue']['data_size_mean'] = p.job_data_size
    config['job_ue']['compute_size_mean'] = p.job_compute_size

    config['reward'].update({
        'delay_penalty': -2.0,
        'sync_base_reward': 10.0,
        'discount_factor': 0.70,
        'e2e_delay_threshold': 2.0,
    })
    return config


def low_traffic_config() -> Dict[str, Any]:
    return dynamic_traffic_config('low')


def medium_traffic_config() -> Dict[str, Any]:
    return dynamic_traffic_config('medium')


def high_traffic_config() -> Dict[str, Any]:
    return dynamic_traffic_config('high')


def ramp_up_traffic_config() -> Dict[str, Any]:
    return dynamic_traffic_config('ramp_up')


def ramp_down_traffic_config() -> Dict[str, Any]:
    return dynamic_traffic_config('ramp_down')


def burst_traffic_config() -> Dict[str, Any]:
    return dynamic_traffic_config('burst')

# ---------------------------------------------------------------------------
# Clean dynamic experiment setup
# ---------------------------------------------------------------------------

# Change this when you want another reward, number of timesteps, or episodes.
DYNAMIC_DEFAULTS = {
    "reward_profile": "balanced_high_load",
    "train_timesteps": 300_000,
    "eval_episodes": 50,
    "seed": 5555,

    "model_tag": "60ue_35s_300k_diffRP",

    # old/stable PPO hyperparameters
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.20,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_epochs": 10,
}

# Change this for the single training point.
# Example: train on 50 UE / 25 sensors, then evaluate the sweep below.
DYNAMIC_TRAINING: Dict[str, Any] = {
    "num_ues": 60,
    "num_sensors": 35,
    "max_steps": 100,
    "compute_capacity": 800,
    "bandwidth": 600e6,
    "bs_positions": [(100.0, 100.0)],
}

# No min_resource_share here. Only the reward parameters you actually compare.
DYNAMIC_REWARD_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "delay_penalty": -2.0,
        "sync_base_reward": 10.0,
        "discount_factor": 0.70,
        "e2e_delay_threshold": 2.0,
    },
        "balanced_high_load": {
        "delay_penalty": -2.5,
        "sync_base_reward": 12.0,
        "discount_factor": 0.75,
        "e2e_delay_threshold": 2.0,
    },
    "delay_focused": {
        "delay_penalty": -4.0,
        "sync_base_reward": 10.0,
        "discount_factor": 0.70,
        "e2e_delay_threshold": 1.5,
    },
    "sync_focused": {
        "delay_penalty": -2.0,
        "sync_base_reward": 15.0,
        "discount_factor": 0.85,
        "e2e_delay_threshold": 2.0,
    },
}

# Change only this block to choose the evaluation sweep.
# mode options:
#   "fixed_sensors" -> UEs change, sensors stay fixed
#   "fixed_ues"     -> sensors change, UEs stay fixed
#   "proportional"  -> UEs and sensors increase together using sensor_to_ue_ratio
DYNAMIC_SWEEP: Dict[str, Any] = {
    "name": "s2_until_80ues_fixed_25sensors_300k",
    "mode": "fixed_sensors",

    # Used when mode = "fixed_sensors"
    "ue_start": 10,
    "ue_stop": 80,
    "ue_step": 5,
    "fixed_sensors": 25,

    # Used when mode = "fixed_ues"
    "fixed_ues": 25,
    "sensor_start": 10,
    "sensor_stop": 80,
    "sensor_step": 5,

    # Used when mode = "proportional"
    "prop_ue_start": 10,
    "prop_ue_stop": 80,
    "prop_ue_step": 10,
    "sensor_to_ue_ratio": 0.50,
}


def apply_dynamic_base(config: Dict[str, Any], num_ues: int, num_sensors: int) -> Dict[str, Any]:
    """Apply the common single-BS settings used for training and sweep evaluation."""
    config['environment']['num_ues'] = int(num_ues)
    config['environment']['num_sensors'] = int(num_sensors)
    config['environment']['max_steps'] = int(DYNAMIC_TRAINING['max_steps'])
    config['bs']['positions'] = list(DYNAMIC_TRAINING['bs_positions'])
    config['bs']['compute_capacity'] = DYNAMIC_TRAINING['compute_capacity']
    config['bs']['bandwidth'] = DYNAMIC_TRAINING['bandwidth']
    return config


def apply_reward_profile(config: Dict[str, Any], reward_profile: str | None = None) -> Dict[str, Any]:
    """Apply one named reward profile from DYNAMIC_REWARD_PROFILES."""
    profile_name = reward_profile or DYNAMIC_DEFAULTS['reward_profile']
    if profile_name not in DYNAMIC_REWARD_PROFILES:
        available = ', '.join(DYNAMIC_REWARD_PROFILES)
        raise ValueError(f"Unknown reward_profile='{profile_name}'. Available: {available}")
    config['reward'].update(DYNAMIC_REWARD_PROFILES[profile_name])
    return config


def dynamic_training_config(reward_profile: str | None = None) -> Dict[str, Any]:
    """One training scenario controlled by DYNAMIC_TRAINING."""
    config = default_config()
    config = apply_dynamic_base(
        config,
        num_ues=DYNAMIC_TRAINING['num_ues'],
        num_sensors=DYNAMIC_TRAINING['num_sensors'],
    )
    config = apply_reward_profile(config, reward_profile)
    return config


def dynamic_eval_config(num_ues: int, num_sensors: int, reward_profile: str | None = None) -> Dict[str, Any]:
    """Create one evaluation scenario from numbers generated by dynamic_sweep_points()."""
    config = default_config()
    config = apply_dynamic_base(config, num_ues=num_ues, num_sensors=num_sensors)
    config = apply_reward_profile(config, reward_profile)
    return config


def dynamic_sweep_points(sweep: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Generate all evaluation points from one DYNAMIC_SWEEP block."""
    sweep = sweep or DYNAMIC_SWEEP
    mode = sweep['mode']
    rows: List[Dict[str, Any]] = []

    if mode == "fixed_sensors":
        fixed_sensors = int(sweep['fixed_sensors'])
        values = range(int(sweep['ue_start']), int(sweep['ue_stop']) + 1, int(sweep['ue_step']))
        for idx, num_ues in enumerate(values):
            rows.append(_make_sweep_row(idx, mode, int(num_ues), fixed_sensors))
        return rows

    if mode == "fixed_ues":
        fixed_ues = int(sweep['fixed_ues'])
        values = range(int(sweep['sensor_start']), int(sweep['sensor_stop']) + 1, int(sweep['sensor_step']))
        for idx, num_sensors in enumerate(values):
            rows.append(_make_sweep_row(idx, mode, fixed_ues, int(num_sensors)))
        return rows

    if mode == "proportional":
        ratio = float(sweep['sensor_to_ue_ratio'])
        values = range(int(sweep['prop_ue_start']), int(sweep['prop_ue_stop']) + 1, int(sweep['prop_ue_step']))
        for idx, num_ues in enumerate(values):
            num_sensors = max(1, int(round(num_ues * ratio)))
            rows.append(_make_sweep_row(idx, mode, int(num_ues), num_sensors))
        return rows

    raise ValueError("DYNAMIC_SWEEP['mode'] must be: fixed_sensors, fixed_ues, or proportional")


def _make_sweep_row(index: int, mode: str, num_ues: int, num_sensors: int) -> Dict[str, Any]:
    total_devices = int(num_ues + num_sensors)
    return {
        "index": index,
        "mode": mode,
        "scenario_name": f"eval_{num_ues}ue_{num_sensors}s",
        "num_ues": int(num_ues),
        "num_sensors": int(num_sensors),
        "total_devices": total_devices,
        "target_split": float(num_ues / total_devices) if total_devices else 0.0,
        "x_label": f"{num_ues} UE\n{num_sensors} S\nT={total_devices}",
    }


def dynamic_model_filename(reward_profile: str | None = None, train_timesteps: int | None = None) -> str:
    """Consistent model filename used by train_dynamic_model.py and dynamic_sweep.py."""
    reward_profile = reward_profile or DYNAMIC_DEFAULTS['reward_profile']
    train_timesteps = int(train_timesteps or DYNAMIC_DEFAULTS['train_timesteps'])
    steps_k = train_timesteps // 1000

    model_tag = str(DYNAMIC_DEFAULTS.get("model_tag", "")).strip()

    name = (
        f"ppo_dynamic_{reward_profile}_"
        f"{DYNAMIC_TRAINING['num_ues']}ue_{DYNAMIC_TRAINING['num_sensors']}s_"
        f"{steps_k}k"
    )

    if model_tag:
        name += f"_{model_tag}"

    return name + ".zip"


def reward_profile_label(reward_profile: str | None = None) -> str:
    """Short text used on top of plots."""
    reward_profile = reward_profile or DYNAMIC_DEFAULTS['reward_profile']
    reward = DYNAMIC_REWARD_PROFILES[reward_profile]
    return (
        f"Reward: delay_penalty={reward['delay_penalty']}, "
        f"sync_base_reward={reward['sync_base_reward']}, "
        f"discount_factor={reward['discount_factor']}, "
        f"D_th={reward['e2e_delay_threshold']}"
    )