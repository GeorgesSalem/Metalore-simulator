from __future__ import annotations

"""
Train ONE general PPO policy for MetaLore with dynamically changing UE/sensor counts.

Final Big20 design
------------------
1. Uses one fixed reward configuration (default command: eta=0.8, C2=-1).
2. Builds one pool of 100 UEs and 50 sensors, but normally trains on 4..80 UEs
   and 4..40 sensors. Active devices arrive/depart every 50 simulator steps.
3. Samples mostly supported/near-capacity loads, with controlled overload phases.
4. Uses multiple simulator workers to feed ONE shared PPO policy. It does not
   create one model per scenario.
5. Saves periodic checkpoints, supports automatic resume, evaluates every
   checkpoint, selects the earliest near-best policy, and creates report-ready
   CSV/JSON files and plots (capacity, training, KPIs, queues, actions, dynamics).
6. Writes all outputs under one directory and creates a compressed final archive.

Place this file in the project root, next to setup.py, then run it from there.
"""

import argparse
import os
import platform
import json
import math
import re
import shutil
import signal
import socket
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from metalore.config.default import default_config
from metalore.handlers.smart_city import SmartCityHandler
from metalore.scenarios import SingleCellEnv


# -----------------------------------------------------------------------------
# Fixed project settings
# -----------------------------------------------------------------------------

BS_POSITION = (100.0, 100.0)
BS_BANDWIDTH_HZ = 600e6          # 600 MHz
BS_COMPUTE_CAPACITY = 800.0
MAP_WIDTH = 200.0
MAP_HEIGHT = 200.0

# The simulator creates a fixed entity pool once. Training normally uses a
# smaller dynamic range, while the full pool remains available for stress tests.
POOL_MAX_UE = 100
POOL_MAX_SENSOR = 50

MIN_UE = 4
TRAIN_MAX_UE = 80
UE_STEP = 2
MIN_SENSOR = 4
TRAIN_MAX_SENSOR = 40
SENSOR_STEP = 2

SYNC_BASE_REWARD = 10.0
E2E_DELAY_THRESHOLD = 2.0
MIN_RESOURCE_SHARE = 0.05        # PPO action range becomes [0.05, 0.95]

DEFAULT_TRAIN_TIMESTEPS = 1_000_000
DEFAULT_EPISODE_STEPS = 1_000
DEFAULT_LOAD_HOLD_STEPS = 50
DEFAULT_NUM_ENVS = 24
DEFAULT_CHECKPOINT_EVERY = 100_000
DEFAULT_EVAL_EPISODES_PER_TARGET = 3
DEFAULT_SEED = 5555

# Dynamic training curriculum. The exact supported boundary is recalculated from
# the dataset/capacity report; these probabilities describe how often PPO sees
# normal, moderate-overload, and strong-overload phases.
SUPPORTED_PHASE_PROBABILITY = 0.70
MODERATE_PHASE_PROBABILITY = 0.20
STRONG_PHASE_PROBABILITY = 0.10
MODERATE_OVERLOAD_TOTAL_MAX = 100

# Static dataset acceptance criteria used only to estimate the practical support
# frontier. They do not change the PPO reward.
SUPPORT_COMPLETION_MIN = 0.90
SUPPORT_AORI_MAX = 4.0
SUPPORT_AOSI_MAX = 4.0


# -----------------------------------------------------------------------------
# Dataset analysis: choose one fixed eta/C2 pair
# -----------------------------------------------------------------------------

def _minmax_penalty(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = float(values.min())
    hi = float(values.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.0, index=series.index)
    if higher_is_better:
        return (hi - values) / (hi - lo)
    return (values - lo) / (hi - lo)


def choose_fixed_reward_pair(
    dataset_path: Path,
    output_dir: Path,
    eta_override: Optional[float] = None,
    c2_override: Optional[float] = None,
) -> Tuple[float, float, pd.DataFrame, Dict[str, Any]]:
    """
    Select one pair across all complete grid scenarios.

    The score is normalized inside each (num_ues, num_sensors) scenario so that
    large-load scenarios do not dominate simply because their KPI values are larger.

    Weights:
      55% completion rate
      20% AoRI
      15% AoSI
      10% throughput

    A 90th-percentile penalty is added so the selected pair is not only good on
    average but also reasonably robust across difficult loads.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path.resolve()}")

    df = pd.read_csv(dataset_path)
    required = {
        "num_ues", "num_sensors", "eta", "c2", "job_completion_rate",
        "mean_aori", "mean_aosi", "throughput_jobs_per_step",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("done")].copy()

    df = df.drop_duplicates(
        subset=["num_ues", "num_sensors", "eta", "c2"], keep="last"
    ).copy()

    # Exclude severely incomplete scenarios. In the supplied data, 100 UE / 50 S
    # has only four completed reward pairs, while the other scenarios have 45.
    scenario_coverage = df.groupby(["num_ues", "num_sensors"]).size()
    max_coverage = int(scenario_coverage.max())
    minimum_coverage = max(2, int(math.ceil(0.90 * max_coverage)))
    complete_scenarios = set(
        scenario_coverage[scenario_coverage >= minimum_coverage].index.tolist()
    )
    mask = [
        (int(u), int(s)) in complete_scenarios
        for u, s in zip(df["num_ues"], df["num_sensors"])
    ]
    ranking_df = df.loc[mask].copy()

    scored_groups: List[pd.DataFrame] = []
    for _, group in ranking_df.groupby(["num_ues", "num_sensors"], sort=False):
        g = group.copy()
        g["penalty_completion"] = _minmax_penalty(g["job_completion_rate"], True)
        g["penalty_aori"] = _minmax_penalty(g["mean_aori"], False)
        g["penalty_aosi"] = _minmax_penalty(g["mean_aosi"], False)
        g["penalty_throughput"] = _minmax_penalty(g["throughput_jobs_per_step"], True)
        g["scenario_score"] = (
            0.55 * g["penalty_completion"]
            + 0.20 * g["penalty_aori"]
            + 0.15 * g["penalty_aosi"]
            + 0.10 * g["penalty_throughput"]
        )
        scored_groups.append(g)

    scored = pd.concat(scored_groups, ignore_index=True)
    ranking = (
        scored.groupby(["eta", "c2"], as_index=False)
        .agg(
            scenarios=("scenario_score", "size"),
            mean_scenario_score=("scenario_score", "mean"),
            p90_scenario_score=("scenario_score", lambda x: float(np.quantile(x, 0.90))),
            mean_completion=("job_completion_rate", "mean"),
            mean_aori=("mean_aori", "mean"),
            mean_aosi=("mean_aosi", "mean"),
            mean_throughput=("throughput_jobs_per_step", "mean"),
            mean_convergence_timestep=(
                "reward_convergence_timestep",
                "mean" if "reward_convergence_timestep" in scored.columns else "size",
            ),
        )
    )
    ranking["robust_score"] = (
        ranking["mean_scenario_score"] + 0.25 * ranking["p90_scenario_score"]
    )
    ranking = ranking.sort_values(
        ["robust_score", "mean_completion", "mean_aori", "mean_aosi"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "reward_pair_ranking.csv", index=False)

    if (eta_override is None) ^ (c2_override is None):
        raise ValueError("Use both --eta and --c2 together, or omit both.")

    if eta_override is not None and c2_override is not None:
        eta = float(eta_override)
        c2 = float(c2_override)
        matching = ranking[
            np.isclose(ranking["eta"], eta) & np.isclose(ranking["c2"], c2)
        ]
        if matching.empty:
            raise ValueError(f"Requested eta={eta}, c2={c2} is not in the dataset.")
        selection_source = "manual override"
    else:
        eta = float(ranking.iloc[0]["eta"])
        c2 = float(ranking.iloc[0]["c2"])
        selection_source = "robust dataset ranking"

    pair_rows = df[np.isclose(df["eta"], eta) & np.isclose(df["c2"], c2)].copy()
    pair_rows["meets_support_targets"] = (
        (pair_rows["job_completion_rate"] >= SUPPORT_COMPLETION_MIN)
        & (pair_rows["mean_aori"] <= SUPPORT_AORI_MAX)
        & (pair_rows["mean_aosi"] <= SUPPORT_AOSI_MAX)
    )
    feasible = pair_rows[pair_rows["meets_support_targets"]].copy()

    if feasible.empty:
        empirical_max_total = None
        frontier = pd.DataFrame()
    else:
        feasible["total_devices"] = feasible["num_ues"] + feasible["num_sensors"]
        empirical_max_total = int(feasible["total_devices"].max())
        frontier = (
            feasible.sort_values(["num_sensors", "num_ues"])
            .groupby("num_sensors", as_index=False)
            .tail(1)[
                [
                    "num_ues", "num_sensors", "total_devices",
                    "job_completion_rate", "mean_aori", "mean_aosi",
                ]
            ]
            .sort_values("num_sensors")
        )
        frontier.to_csv(output_dir / "empirical_support_frontier.csv", index=False)

    convergence = pd.to_numeric(
        pair_rows.get("reward_convergence_timestep", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()

    selection = {
        "source": selection_source,
        "eta": eta,
        "c2": c2,
        "dataset_rows_used": int(len(df)),
        "ranking_scenarios_used": int(len(complete_scenarios)),
        "excluded_incomplete_scenarios": int(len(scenario_coverage) - len(complete_scenarios)),
        "empirical_max_supported_total_devices": empirical_max_total,
        "support_criteria": {
            "completion_rate_min": SUPPORT_COMPLETION_MIN,
            "mean_aori_max": SUPPORT_AORI_MAX,
            "mean_aosi_max": SUPPORT_AOSI_MAX,
        },
        "static_convergence_timestep": {
            "median": float(convergence.median()) if not convergence.empty else None,
            "p90": float(convergence.quantile(0.90)) if not convergence.empty else None,
            "p95": float(convergence.quantile(0.95)) if not convergence.empty else None,
            "max": float(convergence.max()) if not convergence.empty else None,
        },
    }
    return eta, c2, df, selection


# -----------------------------------------------------------------------------
# Theoretical capacity report using the formulas from the board
# -----------------------------------------------------------------------------

def estimate_capacity(seed: int = DEFAULT_SEED, samples: int = 50_000) -> Dict[str, Any]:
    cfg = default_config()

    frequency_mhz = float(cfg["bs"]["frequency"])
    tx_power_dbm = float(cfg["bs"]["tx_power"])
    bs_height = float(cfg["bs"]["height"])
    device_height = float(cfg["ue"]["height"])
    noise_w = float(cfg["ue"]["noise"])

    rng = np.random.default_rng(seed)
    xy = rng.uniform([0.0, 0.0], [MAP_WIDTH, MAP_HEIGHT], size=(samples, 2))
    distance_m = np.sqrt((xy[:, 0] - BS_POSITION[0]) ** 2 + (xy[:, 1] - BS_POSITION[1]) ** 2)
    distance_km = distance_m / 1000.0

    correction = (
        0.8
        + (1.1 * np.log10(frequency_mhz) - 0.7) * device_height
        - 1.56 * np.log10(frequency_mhz)
    )
    tmp1 = (
        69.55
        - correction
        + 26.16 * np.log10(frequency_mhz)
        - 13.82 * np.log10(bs_height)
    )
    tmp2 = 44.9 - 6.55 * np.log10(bs_height)
    path_loss_db = tmp1 + tmp2 * np.log10(distance_km + 1e-16)
    received_power_w = 10.0 ** ((tx_power_dbm - path_loss_db) / 10.0)
    snr = received_power_w / noise_w
    spectral_efficiency = np.log2(1.0 + snr)

    ue_job_probability = float(cfg["job_ue"]["generation_probability"])
    ue_data = float(cfg["job_ue"]["data_size_mean"])
    ue_compute = float(cfg["job_ue"]["compute_size_mean"])

    sensor_interval = max(1, int(cfg["sensor"]["update_interval"]))
    sensor_job_rate = 1.0 / sensor_interval
    sensor_data = float(cfg["job_sensor"]["data_size_mean"])
    sensor_compute = float(cfg["job_sensor"]["compute_size_mean"])

    # Offered load per active device per simulator step.
    ue_comm_load = ue_job_probability * ue_data
    sensor_comm_load = sensor_job_rate * sensor_data
    ue_compute_load = ue_job_probability * ue_compute
    sensor_compute_load = sensor_job_rate * sensor_compute

    # Exact farthest map positions are the four corners (distance sqrt(100^2+100^2)).
    corner_distance_km = math.sqrt(
        max(BS_POSITION[0], MAP_WIDTH - BS_POSITION[0]) ** 2
        + max(BS_POSITION[1], MAP_HEIGHT - BS_POSITION[1]) ** 2
    ) / 1000.0
    corner_path_loss_db = tmp1 + tmp2 * math.log10(corner_distance_km + 1e-16)
    corner_received_power = 10.0 ** ((tx_power_dbm - corner_path_loss_db) / 10.0)
    corner_snr = corner_received_power / noise_w
    corner_se = math.log2(1.0 + corner_snr)

    se_values = {
        "minimum_map_corner": float(corner_se),
        "p10": float(np.quantile(spectral_efficiency, 0.10)),
        "mean": float(np.mean(spectral_efficiency)),
        "median": float(np.median(spectral_efficiency)),
    }

    # The channel implementation returns Mbps: bandwidth * log2(1+SNR) / 1e6.
    # Job data sizes are drained numerically against this value, so capacity is
    # reported in the simulator's per-step data units.
    bandwidth_mhz = BS_BANDWIDTH_HZ / 1e6

    def max_equal_load_total(se: float) -> int:
        # In this configuration both expected loads are 70 units/device/step.
        average_device_load = 0.5 * (ue_comm_load + sensor_comm_load)
        return int(math.floor((bandwidth_mhz * se) / average_device_load))

    compute_total_limit = int(
        math.floor(
            BS_COMPUTE_CAPACITY / (0.5 * (ue_compute_load + sensor_compute_load))
        )
    )

    return {
        "formulas": {
            "ue_communication": "X * lambda_u * d_u <= rho_C * B * log2(1 + SNR_u)",
            "sensor_communication": "Y * lambda_s * d_s <= (1-rho_C) * B * log2(1 + SNR_s)",
            "combined_communication_feasibility": "X*lambda_u*d_u/(B*q_u) + Y*lambda_s*d_s/(B*q_s) <= 1, where q=log2(1+SNR)",
            "ue_computation": "X * lambda_u * c_u <= rho_P * F",
            "sensor_computation": "Y * lambda_s * c_s <= (1-rho_P) * F",
            "combined_compute_feasibility": "X*lambda_u*c_u/F + Y*lambda_s*c_s/F <= 1",
            "overall_capacity": "A load pair is supportable only when both combined feasibility inequalities hold.",
        },
        "bs": {
            "bandwidth_hz": BS_BANDWIDTH_HZ,
            "bandwidth_mhz": bandwidth_mhz,
            "compute_capacity": BS_COMPUTE_CAPACITY,
        },
        "offered_load_per_active_device_per_step": {
            "ue_communication": ue_comm_load,
            "sensor_communication": sensor_comm_load,
            "ue_compute": ue_compute_load,
            "sensor_compute": sensor_compute_load,
        },
        "spectral_efficiency_log2_1_plus_snr": se_values,
        "approx_total_device_limits": {
            "communication_worst_position": max_equal_load_total(se_values["minimum_map_corner"]),
            "communication_p10_position": max_equal_load_total(se_values["p10"]),
            "communication_average_position": max_equal_load_total(se_values["mean"]),
            "computation": compute_total_limit,
        },
        "important_note": (
            "The simulator mixes job data-size units with a channel rate returned in Mbps. "
            "Therefore these communication limits are approximate. The empirical queue/KPI "
            "frontier must be used to confirm the actual supported load."
        ),
    }


def create_capacity_boundary_plot(
    capacity: Dict[str, Any],
    output_dir: Path,
    train_max_ues: int,
    train_max_sensors: int,
    pool_max_ues: int,
    pool_max_sensors: int,
) -> None:
    """Plot analytical UE limits versus active sensors."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    limits = capacity["approx_total_device_limits"]
    sensors = np.arange(MIN_SENSOR, pool_max_sensors + 1, SENSOR_STEP)

    boundary = pd.DataFrame({"num_sensors": sensors})

    plt.figure(figsize=(9, 6))
    for key, label in [
        ("communication_worst_position", "Bandwidth limit: worst position"),
        ("communication_p10_position", "Bandwidth limit: conservative p10"),
        ("communication_average_position", "Bandwidth limit: average position"),
        ("computation", "Computation limit"),
    ]:
        total_limit = int(limits[key])
        max_ues = np.maximum(0, total_limit - sensors)
        boundary[f"max_ues_{key}"] = max_ues
        plt.plot(sensors, max_ues, marker="o", markersize=3, label=label)

    boundary.to_csv(output_dir / "capacity_boundary.csv", index=False)

    plt.scatter(
        [train_max_sensors], [train_max_ues], marker="s", s=80,
        label=f"Training maximum ({train_max_ues} UE, {train_max_sensors} S)",
    )
    plt.scatter(
        [pool_max_sensors], [pool_max_ues], marker="x", s=100,
        label=f"Stress test ({pool_max_ues} UE, {pool_max_sensors} S)",
    )
    plt.xlabel("Number of active sensors")
    plt.ylabel("Maximum active UEs")
    plt.title("Analytical BS capacity boundaries")
    plt.xlim(MIN_SENSOR, pool_max_sensors + 2)
    plt.ylim(0, pool_max_ues + 10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "capacity_boundary.png", dpi=180, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Dynamic environment: one pool, active counts increase by two
# -----------------------------------------------------------------------------

class DynamicQueueHandler(SmartCityHandler):
    """Queue-focused fixed-size observation for a general PPO policy."""

    @classmethod
    def observation_space(cls, env) -> spaces.Space:
        low = np.zeros(8, dtype=np.float32)
        high = np.array(
            [1e4, 1e4, 1e4, 1e4, 1.0, 1.0, 1e4, 1e4],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @classmethod
    def observation(cls, env) -> np.ndarray:
        active_ues = max(1, len(env.active_ues))
        active_sensors = max(1, len(env.active_sensors))

        ue_tx_q = sum(ue.tx_queue.length for ue in env.active_ues)
        sensor_tx_q = sum(sensor.tx_queue.length for sensor in env.active_sensors)
        ue_proc_q = sum(bs.proc_queues["UE"].length for bs in env.stations.values())
        sensor_proc_q = sum(bs.proc_queues["SENSOR"].length for bs in env.stations.values())

        limits = env.config.get("dynamic_load", {})
        max_ues = max(1, int(limits.get("max_ues", env.num_ues)))
        max_sensors = max(1, int(limits.get("max_sensors", env.num_sensors)))

        latest_aori = env.metrics.latest("mean_aori") or 0.0
        latest_aosi = env.metrics.latest("mean_aosi") or 0.0

        return np.array(
            [
                ue_tx_q / active_ues,
                sensor_tx_q / active_sensors,
                ue_proc_q / active_ues,
                sensor_proc_q / active_sensors,
                len(env.active_ues) / max_ues,
                len(env.active_sensors) / max_sensors,
                latest_aori,
                latest_aosi,
            ],
            dtype=np.float32,
        )


class DynamicLoadSingleCellEnv(SingleCellEnv):
    """
    One fixed entity pool and one PPO policy, with piecewise-constant dynamic load.

    During training, every ``load_hold_steps`` the active UE/sensor counts are
    sampled from a curriculum:
      * about 70% supported/near-capacity phases,
      * about 20% moderate overload phases,
      * about 10% strong overload phases.

    Counts are always chosen on the requested step-of-two grid and may increase or
    decrease between phases. This is important: PPO learns both congestion growth
    and queue recovery, instead of seeing only one monotonic ramp.

    ``fixed_target`` is used only for deterministic checkpoint evaluation. The
    episode ramps toward that target and then holds it long enough to observe the
    resulting queues.
    """

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        super().__init__(config=config, render_mode=render_mode)
        dcfg = self.config.get("dynamic_load", {})

        self.pool_max_ues = int(dcfg.get("pool_max_ues", POOL_MAX_UE))
        self.pool_max_sensors = int(dcfg.get("pool_max_sensors", POOL_MAX_SENSOR))
        self.train_max_ues = int(dcfg.get("train_max_ues", TRAIN_MAX_UE))
        self.train_max_sensors = int(dcfg.get("train_max_sensors", TRAIN_MAX_SENSOR))
        self.load_hold_steps = max(1, int(dcfg.get("load_hold_steps", DEFAULT_LOAD_HOLD_STEPS)))

        self.ue_values = list(range(MIN_UE, self.train_max_ues + 1, UE_STEP))
        self.sensor_values = list(range(MIN_SENSOR, self.train_max_sensors + 1, SENSOR_STEP))

        self.support_total = int(dcfg.get("support_total", 80))
        self.moderate_total_max = int(
            dcfg.get("moderate_total_max", MODERATE_OVERLOAD_TOTAL_MAX)
        )
        self.phase_probabilities = np.asarray(
            dcfg.get(
                "phase_probabilities",
                [
                    SUPPORTED_PHASE_PROBABILITY,
                    MODERATE_PHASE_PROBABILITY,
                    STRONG_PHASE_PROBABILITY,
                ],
            ),
            dtype=float,
        )
        self.phase_probabilities = self.phase_probabilities / self.phase_probabilities.sum()

        self.fixed_target = dcfg.get("fixed_target")
        self.fixed_schedule = dcfg.get("fixed_schedule")
        self.clear_departing_tx_queues = bool(
            dcfg.get("clear_departing_tx_queues", True)
        )
        self.schedule_rng = np.random.default_rng(int(self.seed) + 900_001)
        self.entity_rng = np.random.default_rng(int(self.seed) + 900_101)
        self.load_schedule: List[Tuple[int, int]] = []
        self._active_ue_ids: set[int] = set()
        self._active_sensor_ids: set[int] = set()

        all_pairs = [(u, s) for u in self.ue_values for s in self.sensor_values]
        max_training_total = self.train_max_ues + self.train_max_sensors
        moderate_max = min(self.moderate_total_max, max_training_total)

        self.region_pairs: Dict[str, List[Tuple[int, int]]] = {
            "supported": [p for p in all_pairs if sum(p) <= self.support_total],
            "moderate": [
                p for p in all_pairs
                if self.support_total < sum(p) <= moderate_max
            ],
            "strong": [p for p in all_pairs if sum(p) > moderate_max],
        }
        # Graceful fallback if a region is empty after a user changes ranges.
        for region in ("supported", "moderate", "strong"):
            if not self.region_pairs[region]:
                self.region_pairs[region] = all_pairs

    def _sample_training_pair(self, previous: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        region = str(
            self.schedule_rng.choice(
                ["supported", "moderate", "strong"],
                p=self.phase_probabilities,
            )
        )
        pairs = self.region_pairs[region]
        pair = pairs[int(self.schedule_rng.integers(0, len(pairs)))]

        # Avoid repeating exactly the same load in two adjacent phases when possible.
        if previous is not None and len(pairs) > 1:
            for _ in range(20):
                if pair != previous:
                    break
                pair = pairs[int(self.schedule_rng.integers(0, len(pairs)))]
        return int(pair[0]), int(pair[1])

    @staticmethod
    def _nearest_grid_value(value: float, minimum: int, maximum: int, step: int) -> int:
        rounded = minimum + int(round((value - minimum) / step)) * step
        return int(np.clip(rounded, minimum, maximum))

    def _build_fixed_target_schedule(self, target_u: int, target_s: int) -> List[Tuple[int, int]]:
        """Deterministic ramp for validation, followed by a long target hold."""
        target_u = int(np.clip(target_u, MIN_UE, self.pool_max_ues))
        target_s = int(np.clip(target_s, MIN_SENSOR, self.pool_max_sensors))
        phases = max(1, math.ceil(self.EP_MAX_TIME / self.load_hold_steps))
        ramp_phases = max(2, int(round(0.40 * phases)))

        phase_pairs: List[Tuple[int, int]] = []
        for phase in range(phases):
            if phase < ramp_phases:
                fraction = phase / max(1, ramp_phases - 1)
                u = self._nearest_grid_value(
                    MIN_UE + fraction * (target_u - MIN_UE),
                    MIN_UE,
                    self.pool_max_ues,
                    UE_STEP,
                )
                s = self._nearest_grid_value(
                    MIN_SENSOR + fraction * (target_s - MIN_SENSOR),
                    MIN_SENSOR,
                    self.pool_max_sensors,
                    SENSOR_STEP,
                )
                phase_pairs.append((u, s))
            else:
                phase_pairs.append((target_u, target_s))

        return [
            phase_pairs[min(t // self.load_hold_steps, len(phase_pairs) - 1)]
            for t in range(self.EP_MAX_TIME)
        ]

    def _build_training_schedule(self) -> List[Tuple[int, int]]:
        phases = max(1, math.ceil(self.EP_MAX_TIME / self.load_hold_steps))
        phase_pairs: List[Tuple[int, int]] = []
        previous: Optional[Tuple[int, int]] = None
        for _ in range(phases):
            previous = self._sample_training_pair(previous)
            phase_pairs.append(previous)

        return [
            phase_pairs[min(t // self.load_hold_steps, len(phase_pairs) - 1)]
            for t in range(self.EP_MAX_TIME)
        ]

    def _build_explicit_schedule(self) -> List[Tuple[int, int]]:
        """Expand a list of phase pairs into one pair per simulator step."""
        phases = [
            (int(pair[0]), int(pair[1]))
            for pair in (self.fixed_schedule or [])
        ]
        if not phases:
            raise ValueError("fixed_schedule is empty")
        for u, s in phases:
            if not (MIN_UE <= u <= self.pool_max_ues):
                raise ValueError(f"Invalid fixed-schedule UE count: {u}")
            if not (MIN_SENSOR <= s <= self.pool_max_sensors):
                raise ValueError(f"Invalid fixed-schedule sensor count: {s}")
        required_phases = max(1, math.ceil(self.EP_MAX_TIME / self.load_hold_steps))
        if len(phases) < required_phases:
            repeats = math.ceil(required_phases / len(phases))
            phases = (phases * repeats)[:required_phases]
        else:
            phases = phases[:required_phases]
        return [
            phases[min(t // self.load_hold_steps, len(phases) - 1)]
            for t in range(self.EP_MAX_TIME)
        ]

    def _build_schedule(self) -> List[Tuple[int, int]]:
        if self.fixed_schedule is not None:
            return self._build_explicit_schedule()
        if self.fixed_target is not None:
            return self._build_fixed_target_schedule(
                int(self.fixed_target[0]), int(self.fixed_target[1])
            )
        return self._build_training_schedule()

    def _resize_active_ids(
        self,
        current: set[int],
        all_ids: Sequence[int],
        target_count: int,
        entity_lookup: Dict[int, Any],
        force_new: bool = False,
    ) -> set[int]:
        """Add/remove random devices while preserving continuing devices."""
        all_ids_set = set(int(x) for x in all_ids)
        if force_new:
            current = set()
        else:
            current = set(current).intersection(all_ids_set)

        if len(current) > target_count:
            departing = self.entity_rng.choice(
                sorted(current), size=len(current) - target_count, replace=False
            ).tolist()
            for entity_id in departing:
                current.remove(int(entity_id))
                if self.clear_departing_tx_queues:
                    entity_lookup[int(entity_id)].reset_queue()

        if len(current) < target_count:
            inactive = sorted(all_ids_set.difference(current))
            arriving = self.entity_rng.choice(
                inactive, size=target_count - len(current), replace=False
            ).tolist()
            current.update(int(entity_id) for entity_id in arriving)

        return current

    def _apply_active_counts(
        self,
        num_ues: int,
        num_sensors: int,
        *,
        force_new: bool = False,
    ) -> None:
        if not (MIN_UE <= num_ues <= self.pool_max_ues):
            raise ValueError(f"Invalid active UE count: {num_ues}")
        if not (MIN_SENSOR <= num_sensors <= self.pool_max_sensors):
            raise ValueError(f"Invalid active sensor count: {num_sensors}")

        self._active_ue_ids = self._resize_active_ids(
            self._active_ue_ids,
            list(self.users.keys()),
            int(num_ues),
            self.users,
            force_new=force_new,
        )
        self._active_sensor_ids = self._resize_active_ids(
            self._active_sensor_ids,
            list(self.sensors.keys()),
            int(num_sensors),
            self.sensors,
            force_new=force_new,
        )

        self.active_ues = sorted(
            [self.users[entity_id] for entity_id in self._active_ue_ids],
            key=lambda entity: entity.id,
        )
        self.active_sensors = sorted(
            [self.sensors[entity_id] for entity_id in self._active_sensor_ids],
            key=lambda entity: entity.id,
        )
        active_users = {ue.id: ue for ue in self.active_ues}
        active_sensors = {sensor.id: sensor for sensor in self.active_sensors}
        self.association.update_association(self.stations, active_users, active_sensors)
        self.validate_connections()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.load_schedule = self._build_schedule()
        self._active_ue_ids = set()
        self._active_sensor_ids = set()
        self._apply_active_counts(*self.load_schedule[0], force_new=True)
        obs = self.handler.observation(self)
        info = self.handler.info(self)
        info["dynamic_target_ues"] = int(max(u for u, _ in self.load_schedule))
        info["dynamic_target_sensors"] = int(max(s for _, s in self.load_schedule))
        info["load_hold_steps"] = self.load_hold_steps
        return obs, info

    def step(self, actions):
        # Apply the scheduled active population before job generation and service.
        current_index = min(int(self.time), len(self.load_schedule) - 1)
        self._apply_active_counts(*self.load_schedule[current_index])

        obs, reward, terminated, truncated, info = super().step(actions)

        # MetaLore's base step rebuilds active sets from the static arrival model.
        # Restore the next dynamic population and return the matching observation.
        if not (terminated or truncated):
            next_index = min(int(self.time), len(self.load_schedule) - 1)
            self._apply_active_counts(*self.load_schedule[next_index])
            obs = self.handler.observation(self)
            info.update(self.handler.info(self))

        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------
# Environment/config helpers
# -----------------------------------------------------------------------------

def build_dynamic_config(
    *,
    eta: float,
    c2: float,
    episode_steps: int,
    load_hold_steps: int,
    seed: int,
    support_total: int,
    train_max_ues: int,
    train_max_sensors: int,
    pool_max_ues: int,
    pool_max_sensors: int,
    fixed_target: Optional[Tuple[int, int]] = None,
    fixed_schedule: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    config = default_config()
    config["environment"].update(
        {
            "width": MAP_WIDTH,
            "height": MAP_HEIGHT,
            "num_ues": int(pool_max_ues),
            "num_sensors": int(pool_max_sensors),
            "max_steps": int(episode_steps),
            "seed": int(seed),
            "reset_rng_episode": False,
            "handler": DynamicQueueHandler,
        }
    )
    config["bs"].update(
        {
            "positions": [BS_POSITION],
            "bandwidth": BS_BANDWIDTH_HZ,
            "compute_capacity": BS_COMPUTE_CAPACITY,
        }
    )
    config["reward"].update(
        {
            "delay_penalty": float(c2),
            "sync_base_reward": SYNC_BASE_REWARD,
            "discount_factor": float(eta),
            "e2e_delay_threshold": E2E_DELAY_THRESHOLD,
            "min_resource_share": MIN_RESOURCE_SHARE,
        }
    )
    config["dynamic_load"] = {
        "pool_max_ues": int(pool_max_ues),
        "pool_max_sensors": int(pool_max_sensors),
        "train_max_ues": int(train_max_ues),
        "train_max_sensors": int(train_max_sensors),
        "load_hold_steps": int(load_hold_steps),
        "support_total": int(support_total),
        "moderate_total_max": int(MODERATE_OVERLOAD_TOTAL_MAX),
        "phase_probabilities": [
            SUPPORTED_PHASE_PROBABILITY,
            MODERATE_PHASE_PROBABILITY,
            STRONG_PHASE_PROBABILITY,
        ],
        "fixed_target": list(fixed_target) if fixed_target is not None else None,
        "fixed_schedule": [list(pair) for pair in fixed_schedule]
        if fixed_schedule is not None
        else None,
        "clear_departing_tx_queues": True,
    }
    return config


def make_env_factory(
    base_config: Dict[str, Any], rank: int, monitor_dir: Path, run_tag: str
) -> Callable[[], DynamicLoadSingleCellEnv]:
    def _init():
        cfg = deepcopy(base_config)
        cfg["environment"]["seed"] = int(base_config["environment"]["seed"]) + rank * 10_003
        env = DynamicLoadSingleCellEnv(config=cfg)
        monitor_file = monitor_dir / f"train_env_{rank}_{run_tag}_monitor.csv"
        return Monitor(env, filename=str(monitor_file))

    return _init


# -----------------------------------------------------------------------------
# Checkpoint evaluation and best-timestep selection
# -----------------------------------------------------------------------------

def _safe_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else fallback
    except (TypeError, ValueError):
        return fallback


def _queue_series(env: DynamicLoadSingleCellEnv) -> np.ndarray:
    n = env.metrics.num_steps
    if n == 0:
        return np.zeros(1, dtype=float)

    ue_tx = np.asarray(env.metrics.step_totals["ue_tx_queue_jobs"], dtype=float)
    sensor_tx = np.asarray(env.metrics.step_totals["sensor_tx_queue_jobs"], dtype=float)

    ue_proc = np.zeros(n, dtype=float)
    sensor_proc = np.zeros(n, dtype=float)
    for values in env.metrics.step_per_bs["ue_proc_queue_jobs"].values():
        ue_proc += np.asarray(values[:n], dtype=float)
    for values in env.metrics.step_per_bs["sensor_proc_queue_jobs"].values():
        sensor_proc += np.asarray(values[:n], dtype=float)

    return ue_tx + sensor_tx + ue_proc + sensor_proc


def evaluate_checkpoint(
    model_path: Path,
    eta: float,
    c2: float,
    episode_steps: int,
    load_hold_steps: int,
    support_total: int,
    train_max_ues: int,
    train_max_sensors: int,
    pool_max_ues: int,
    pool_max_sensors: int,
    eval_episodes_per_target: int,
    seed: int,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Feasible and overload validation paths. Each episode ramps from 4/4 to target.
    targets: List[Tuple[int, int, float, str]] = [
        (20, 10, 1.00, "low_supported"),
        (40, 20, 1.00, "medium_supported"),
        (60, 20, 1.00, "ue_heavy_boundary"),
        (40, 40, 1.00, "balanced_boundary"),
        (30, 50, 1.00, "sensor_heavy_boundary"),
        (70, 20, 0.75, "moderate_overload"),
        (80, 40, 0.50, "strong_overload"),
        (100, 50, 0.20, "severe_stress_test"),
    ]

    model = PPO.load(str(model_path), device=device)
    rows: List[Dict[str, Any]] = []

    for target_index, (target_u, target_s, weight, label) in enumerate(targets):
        cfg = build_dynamic_config(
            eta=eta,
            c2=c2,
            episode_steps=episode_steps,
            load_hold_steps=load_hold_steps,
            seed=seed + target_index * 1009,
            support_total=support_total,
            train_max_ues=train_max_ues,
            train_max_sensors=train_max_sensors,
            pool_max_ues=pool_max_ues,
            pool_max_sensors=pool_max_sensors,
            fixed_target=(target_u, target_s),
        )
        env = DynamicLoadSingleCellEnv(config=cfg)

        for episode in range(eval_episodes_per_target):
            obs, _ = env.reset()
            done = False
            rewards: List[float] = []
            bw_actions: List[float] = []
            comp_actions: List[float] = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                rewards.append(float(reward))
                bw_actions.append(float(action[0]))
                comp_actions.append(float(action[1]))

            summary = env.metrics.finalize(env.job_tracker)
            queue = _queue_series(env)
            tail_start = max(0, int(0.60 * len(queue)))
            tail = queue[tail_start:]
            if len(tail) >= 2:
                queue_slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
            else:
                queue_slope = 0.0

            completion = _safe_float(summary.get("job_completion_rate"), 0.0)
            aori = _safe_float(summary.get("mean_aori"), float(episode_steps * 2))
            aosi = _safe_float(summary.get("mean_aosi"), float(episode_steps * 2))

            # Lower is better. Completion is the first priority; then AoRI/AoSI;
            # then queue growth and queue size.
            completion_gap = max(0.0, SUPPORT_COMPLETION_MIN - completion)
            validation_score = (
                1000.0 * completion_gap
                + 3.0 * aori
                + 2.0 * aosi
                + 2.0 * max(0.0, queue_slope)
                + 0.01 * float(np.mean(queue))
            )

            rows.append(
                {
                    "model_path": str(model_path),
                    "target_label": label,
                    "target_ues": target_u,
                    "target_sensors": target_s,
                    "target_total": target_u + target_s,
                    "weight": weight,
                    "episode": episode + 1,
                    "job_completion_rate": completion,
                    "mean_aori": aori,
                    "mean_aosi": aosi,
                    "mean_total_queue_jobs": float(np.mean(queue)),
                    "final_total_queue_jobs": float(queue[-1]),
                    "tail_queue_growth_per_step": queue_slope,
                    "mean_bw_split_to_ues": float(np.mean(bw_actions)),
                    "mean_comp_split_to_ues": float(np.mean(comp_actions)),
                    "total_reward": float(np.sum(rewards)),
                    "validation_score": validation_score,
                }
            )

        env.close()

    episodes = pd.DataFrame(rows)
    weighted_score = float(
        np.average(episodes["validation_score"], weights=episodes["weight"])
    )
    supported = episodes[episodes["target_total"] <= int(support_total)].copy()
    if supported.empty:
        supported = episodes.copy()

    aggregate = {
        "weighted_validation_score": weighted_score,
        "mean_completion": float(
            np.average(episodes["job_completion_rate"], weights=episodes["weight"])
        ),
        "mean_aori": float(np.average(episodes["mean_aori"], weights=episodes["weight"])),
        "mean_aosi": float(np.average(episodes["mean_aosi"], weights=episodes["weight"])),
        "mean_queue": float(
            np.average(episodes["mean_total_queue_jobs"], weights=episodes["weight"])
        ),
        "mean_positive_queue_slope": float(
            np.average(
                np.maximum(0.0, episodes["tail_queue_growth_per_step"]),
                weights=episodes["weight"],
            )
        ),
        "supported_mean_completion": float(supported["job_completion_rate"].mean()),
        "supported_mean_aori": float(supported["mean_aori"].mean()),
        "supported_mean_aosi": float(supported["mean_aosi"].mean()),
        "supported_mean_queue": float(supported["mean_total_queue_jobs"].mean()),
        "supported_mean_positive_queue_slope": float(
            np.maximum(0.0, supported["tail_queue_growth_per_step"]).mean()
        ),
    }
    return episodes, aggregate


def checkpoint_timestep(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)k", path.name)
    if match:
        return int(match.group(1)) * 1000
    return 10**18


def select_best_checkpoint(
    checkpoints: Sequence[Path],
    output_dir: Path,
    eta: float,
    c2: float,
    episode_steps: int,
    load_hold_steps: int,
    support_total: int,
    train_max_ues: int,
    train_max_sensors: int,
    pool_max_ues: int,
    pool_max_sensors: int,
    eval_episodes_per_target: int,
    seed: int,
    device: str,
) -> Tuple[Path, int, pd.DataFrame]:
    aggregate_rows: List[Dict[str, Any]] = []
    all_episode_rows: List[pd.DataFrame] = []

    for model_path in sorted(checkpoints, key=checkpoint_timestep):
        steps = checkpoint_timestep(model_path)
        print(f"\nEvaluating checkpoint: {model_path.name} ({steps:,} steps)")
        episodes, aggregate = evaluate_checkpoint(
            model_path=model_path,
            eta=eta,
            c2=c2,
            episode_steps=episode_steps,
            load_hold_steps=load_hold_steps,
            support_total=support_total,
            train_max_ues=train_max_ues,
            train_max_sensors=train_max_sensors,
            pool_max_ues=pool_max_ues,
            pool_max_sensors=pool_max_sensors,
            eval_episodes_per_target=eval_episodes_per_target,
            seed=seed,
            device=device,
        )
        episodes.insert(0, "checkpoint_timesteps", steps)
        all_episode_rows.append(episodes)
        aggregate_rows.append(
            {
                "checkpoint_timesteps": steps,
                "model_path": str(model_path),
                **aggregate,
            }
        )
        print(
            "score=", round(aggregate["weighted_validation_score"], 4),
            "completion=", round(aggregate["mean_completion"], 4),
            "AoRI=", round(aggregate["mean_aori"], 4),
            "AoSI=", round(aggregate["mean_aosi"], 4),
        )

    summary = pd.DataFrame(aggregate_rows).sort_values("checkpoint_timesteps")
    details = pd.concat(all_episode_rows, ignore_index=True)
    summary.to_csv(output_dir / "checkpoint_validation_summary.csv", index=False)
    details.to_csv(output_dir / "checkpoint_validation_episodes.csv", index=False)

    best_score = float(summary["weighted_validation_score"].min())
    score_tolerance = 0.02 * max(1.0, abs(best_score))
    best_supported_completion = float(summary["supported_mean_completion"].max())
    completion_floor = max(
        SUPPORT_COMPLETION_MIN,
        best_supported_completion - 0.02,
    )
    best_supported_slope = float(
        summary["supported_mean_positive_queue_slope"].min()
    )
    slope_tolerance = max(0.02, 0.10 * abs(best_supported_slope))

    eligible = summary[
        (summary["weighted_validation_score"] <= best_score + score_tolerance)
        & (summary["supported_mean_completion"] >= completion_floor)
        & (
            summary["supported_mean_positive_queue_slope"]
            <= best_supported_slope + slope_tolerance
        )
    ].sort_values("checkpoint_timesteps")

    if eligible.empty:
        eligible = summary[
            summary["weighted_validation_score"] <= best_score + score_tolerance
        ].sort_values("checkpoint_timesteps")

    selected = eligible.iloc[0]

    best_path = Path(str(selected["model_path"]))
    best_steps = int(selected["checkpoint_timesteps"])
    return best_path, best_steps, summary


def create_result_plots(output_dir: Path, best_steps: int) -> None:
    """Create report-ready plots from checkpoint validation CSV files."""
    summary_path = output_dir / "checkpoint_validation_summary.csv"
    details_path = output_dir / "checkpoint_validation_episodes.csv"
    if not summary_path.exists() or not details_path.exists():
        print("Plot warning: validation CSV files are missing; plots were skipped.")
        return

    summary = pd.read_csv(summary_path).sort_values("checkpoint_timesteps")
    details = pd.read_csv(details_path)
    best = details[details["checkpoint_timesteps"] == int(best_steps)].copy()
    if best.empty:
        print("Plot warning: selected checkpoint rows were not found.")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(plot_dir / name, dpi=180, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["checkpoint_timesteps"], summary["weighted_validation_score"], marker="o")
    plt.axvline(best_steps, linestyle="--", label=f"selected: {best_steps:,}")
    plt.xlabel("PPO training timesteps")
    plt.ylabel("Weighted validation score (lower is better)")
    plt.title("Checkpoint selection")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("checkpoint_validation_score.png")

    plt.figure(figsize=(8, 5))
    plt.plot(summary["checkpoint_timesteps"], summary["mean_completion"], marker="o")
    plt.axvline(best_steps, linestyle="--")
    plt.xlabel("PPO training timesteps")
    plt.ylabel("Mean completion rate")
    plt.title("Completion rate versus training timesteps")
    plt.grid(True, alpha=0.3)
    save("checkpoint_completion_rate.png")

    plt.figure(figsize=(8, 5))
    plt.plot(summary["checkpoint_timesteps"], summary["mean_aori"], marker="o", label="AoRI")
    plt.plot(summary["checkpoint_timesteps"], summary["mean_aosi"], marker="o", label="AoSI")
    plt.axvline(best_steps, linestyle="--")
    plt.xlabel("PPO training timesteps")
    plt.ylabel("Mean age value")
    plt.title("AoRI and AoSI versus training timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("checkpoint_aori_aosi.png")

    grouped = (
        best.groupby(["target_label", "target_ues", "target_sensors", "target_total"], as_index=False)
        .agg(
            job_completion_rate=("job_completion_rate", "mean"),
            mean_aori=("mean_aori", "mean"),
            mean_aosi=("mean_aosi", "mean"),
            mean_total_queue_jobs=("mean_total_queue_jobs", "mean"),
            tail_queue_growth_per_step=("tail_queue_growth_per_step", "mean"),
            mean_bw_split_to_ues=("mean_bw_split_to_ues", "mean"),
            mean_comp_split_to_ues=("mean_comp_split_to_ues", "mean"),
        )
        .sort_values(["target_total", "target_ues"])
    )
    grouped.to_csv(output_dir / "best_model_capacity_summary.csv", index=False)

    labels = [
        f"{int(u)}U/{int(s)}S"
        for u, s in zip(grouped["target_ues"], grouped["target_sensors"])
    ]
    x = np.arange(len(grouped))

    plt.figure(figsize=(10, 5))
    plt.plot(x, grouped["job_completion_rate"], marker="o")
    plt.axhline(SUPPORT_COMPLETION_MIN, linestyle="--", label="target 0.90")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Completion rate")
    plt.title("Selected PPO: completion across load levels")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("best_model_completion_vs_load.png")

    plt.figure(figsize=(10, 5))
    plt.plot(x, grouped["mean_aori"], marker="o", label="AoRI")
    plt.plot(x, grouped["mean_aosi"], marker="o", label="AoSI")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Mean age value")
    plt.title("Selected PPO: AoRI and AoSI across load levels")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("best_model_aori_aosi_vs_load.png")

    plt.figure(figsize=(10, 5))
    plt.plot(x, grouped["mean_total_queue_jobs"], marker="o", label="mean queue")
    plt.plot(x, grouped["tail_queue_growth_per_step"], marker="o", label="tail queue slope")
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Queue jobs / growth per step")
    plt.title("Selected PPO: queue level and overload growth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("best_model_queue_vs_load.png")

    plt.figure(figsize=(10, 5))
    plt.plot(x, grouped["mean_bw_split_to_ues"], marker="o", label="bandwidth share to UEs")
    plt.plot(x, grouped["mean_comp_split_to_ues"], marker="o", label="compute share to UEs")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean PPO action")
    plt.title("Selected PPO: learned resource splits")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("best_model_resource_splits_vs_load.png")

    print("Plots saved in:", plot_dir)



def create_training_monitor_plots(output_dir: Path) -> None:
    """Aggregate all SB3 Monitor files and plot training reward/episode length."""
    monitor_dir = output_dir / "monitors"
    files = sorted(monitor_dir.glob("*monitor.csv"))
    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            frame = pd.read_csv(file, comment="#")
        except Exception as exc:
            print(f"Monitor warning for {file.name}: {exc}")
            continue
        if not {"r", "l"}.issubset(frame.columns):
            continue
        if "t" not in frame.columns:
            frame["t"] = np.arange(len(frame), dtype=float)
        frame["source"] = file.name
        frames.append(frame)

    if not frames:
        print("Plot warning: no valid monitor files were found.")
        return

    episodes = pd.concat(frames, ignore_index=True)
    episodes = episodes.sort_values("t").reset_index(drop=True)
    episodes["global_completed_timesteps"] = pd.to_numeric(
        episodes["l"], errors="coerce"
    ).fillna(0).cumsum()
    episodes["reward_rolling_50"] = pd.to_numeric(
        episodes["r"], errors="coerce"
    ).rolling(50, min_periods=1).mean()
    episodes["length_rolling_50"] = pd.to_numeric(
        episodes["l"], errors="coerce"
    ).rolling(50, min_periods=1).mean()
    episodes.to_csv(output_dir / "training_monitor_episodes.csv", index=False)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(
        episodes["global_completed_timesteps"],
        episodes["reward_rolling_50"],
    )
    plt.xlabel("Completed training timesteps")
    plt.ylabel("Rolling mean episode reward (50 episodes)")
    plt.title("General PPO training reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "training_reward_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        episodes["global_completed_timesteps"],
        episodes["length_rolling_50"],
    )
    plt.xlabel("Completed training timesteps")
    plt.ylabel("Rolling mean episode length")
    plt.title("Training episode length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "training_episode_length.png", dpi=180, bbox_inches="tight")
    plt.close()


def _latest_bs_metric(env: DynamicLoadSingleCellEnv, metric: str) -> float:
    data = env.metrics.step_per_bs.get(metric, {})
    total = 0.0
    for values in data.values():
        if values:
            total += float(values[-1])
    return total


def evaluate_dynamic_response_trace(
    model_path: Path,
    output_dir: Path,
    eta: float,
    c2: float,
    episode_steps: int,
    load_hold_steps: int,
    support_total: int,
    train_max_ues: int,
    train_max_sensors: int,
    pool_max_ues: int,
    pool_max_sensors: int,
    seed: int,
    device: str,
) -> None:
    """Evaluate one low/capacity/overload/recovery sequence and plot reactions."""
    base_phases: List[Tuple[int, int]] = [
        (20, 10),
        (40, 20),
        (50, 30),
        (60, 20),
        (40, 40),
        (70, 20),
        (80, 20),
        (80, 40),
        (100, 50),
        (80, 40),
        (60, 20),
        (40, 20),
        (20, 10),
        (30, 50),
        (50, 30),
        (70, 20),
        (40, 40),
        (20, 10),
        (80, 40),
        (20, 10),
    ]
    required_phases = max(1, math.ceil(episode_steps / load_hold_steps))
    repeats = math.ceil(required_phases / len(base_phases))
    phases = (base_phases * repeats)[:required_phases]

    cfg = build_dynamic_config(
        eta=eta,
        c2=c2,
        episode_steps=episode_steps,
        load_hold_steps=load_hold_steps,
        seed=seed,
        support_total=support_total,
        train_max_ues=train_max_ues,
        train_max_sensors=train_max_sensors,
        pool_max_ues=pool_max_ues,
        pool_max_sensors=pool_max_sensors,
        fixed_schedule=phases,
    )
    env = DynamicLoadSingleCellEnv(config=cfg)
    model = PPO.load(str(model_path), device=device)

    obs, _ = env.reset()
    rows: List[Dict[str, Any]] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        st = env.metrics.step_totals
        idx = -1
        rows.append(
            {
                "time": int(st["time"][idx]),
                "num_active_ues": int(st["num_active_ues"][idx]),
                "num_active_sensors": int(st["num_active_sensors"][idx]),
                "total_active_devices": int(st["num_active_ues"][idx])
                + int(st["num_active_sensors"][idx]),
                "ue_tx_queue_jobs": float(st["ue_tx_queue_jobs"][idx]),
                "sensor_tx_queue_jobs": float(st["sensor_tx_queue_jobs"][idx]),
                "ue_proc_queue_jobs": _latest_bs_metric(env, "ue_proc_queue_jobs"),
                "sensor_proc_queue_jobs": _latest_bs_metric(env, "sensor_proc_queue_jobs"),
                "jobs_generated": float(st["jobs_generated"][idx]),
                "jobs_transmitted": float(st["jobs_transmitted"][idx]),
                "jobs_processed": float(st["jobs_processed"][idx]),
                "bw_split_to_ues": float(st["bw_split"][idx]),
                "comp_split_to_ues": float(st["comp_split"][idx]),
                "reward": float(reward),
                "mean_aori": st["mean_aori"][idx],
                "mean_aosi": st["mean_aosi"][idx],
            }
        )

    final_summary = env.metrics.finalize(env.job_tracker)
    env.close()

    trace = pd.DataFrame(rows)
    trace["total_queue_jobs"] = (
        trace["ue_tx_queue_jobs"]
        + trace["sensor_tx_queue_jobs"]
        + trace["ue_proc_queue_jobs"]
        + trace["sensor_proc_queue_jobs"]
    )
    trace.to_csv(output_dir / "dynamic_response_trace.csv", index=False)
    with (output_dir / "dynamic_response_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(trace["time"], trace["num_active_ues"], label="Active UEs")
    plt.plot(trace["time"], trace["num_active_sensors"], label="Active sensors")
    plt.plot(trace["time"], trace["total_active_devices"], label="Total devices")
    plt.axhline(support_total, linestyle="--", label=f"Practical capacity: {support_total}")
    plt.xlabel("Simulation timestep")
    plt.ylabel("Active devices")
    plt.title("Dynamic validation load profile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "dynamic_active_devices.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(trace["time"], trace["ue_tx_queue_jobs"], label="UE transmission queue")
    plt.plot(trace["time"], trace["sensor_tx_queue_jobs"], label="Sensor transmission queue")
    plt.plot(trace["time"], trace["ue_proc_queue_jobs"], label="UE processing queue")
    plt.plot(trace["time"], trace["sensor_proc_queue_jobs"], label="Sensor processing queue")
    plt.xlabel("Simulation timestep")
    plt.ylabel("Queued jobs")
    plt.title("Queue response to dynamic load")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "dynamic_queues.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(trace["time"], trace["bw_split_to_ues"], label="Bandwidth share to UEs")
    plt.plot(trace["time"], trace["comp_split_to_ues"], label="Compute share to UEs")
    plt.xlabel("Simulation timestep")
    plt.ylabel("PPO action")
    plt.ylim(0.0, 1.0)
    plt.title("PPO resource-allocation response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "dynamic_resource_splits.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(trace["time"], trace["jobs_generated"], label="Jobs generated")
    plt.plot(trace["time"], trace["jobs_transmitted"], label="Jobs transmitted")
    plt.plot(trace["time"], trace["jobs_processed"], label="Jobs processed")
    plt.xlabel("Simulation timestep")
    plt.ylabel("Jobs per step")
    plt.title("Dynamic job flow")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "dynamic_job_flow.png", dpi=180, bbox_inches="tight")
    plt.close()

    kpi_trace = trace.copy()
    kpi_trace["mean_aori"] = pd.to_numeric(kpi_trace["mean_aori"], errors="coerce")
    kpi_trace["mean_aosi"] = pd.to_numeric(kpi_trace["mean_aosi"], errors="coerce")
    kpi_trace["aori_rolling_20"] = kpi_trace["mean_aori"].rolling(20, min_periods=1).mean()
    kpi_trace["aosi_rolling_20"] = kpi_trace["mean_aosi"].rolling(20, min_periods=1).mean()
    plt.figure(figsize=(11, 5))
    plt.plot(kpi_trace["time"], kpi_trace["aori_rolling_20"], label="AoRI rolling mean")
    plt.plot(kpi_trace["time"], kpi_trace["aosi_rolling_20"], label="AoSI rolling mean")
    plt.xlabel("Simulation timestep")
    plt.ylabel("Age value")
    plt.title("Dynamic AoRI and AoSI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "dynamic_aori_aosi.png", dpi=180, bbox_inches="tight")
    plt.close()


def create_results_archive(output_dir: Path, project_root: Path) -> Path:
    """Create one portable tar.gz containing all final models, tables and plots."""
    backup_dir = project_root / "results_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = backup_dir / f"{output_dir.name}_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    return archive


def write_run_status(output_dir: Path, status: str, **extra: Any) -> None:
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        **extra,
    }
    with (output_dir / "run_status.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def find_latest_resume_model(output_dir: Path) -> Optional[Path]:
    candidates = list((output_dir / "checkpoints").glob("*_steps.zip"))
    candidates += list((output_dir / "models").glob("*final_*_steps.zip"))
    candidates += list((output_dir / "models").glob("emergency_*_steps.zip"))
    if not candidates:
        return None
    return max(candidates, key=checkpoint_timestep)


def install_emergency_signal_handlers(model: PPO, output_dir: Path) -> None:
    """Save one emergency model when OAR sends SIGTERM before walltime."""
    def _handler(signum, _frame):
        try:
            emergency = output_dir / "models" / f"emergency_{model.num_timesteps}_steps"
            print(f"\nReceived signal {signum}; saving emergency model to {emergency}.zip")
            model.save(str(emergency))
            write_run_status(
                output_dir,
                "interrupted",
                signal=int(signum),
                saved_timesteps=int(model.num_timesteps),
                emergency_model=str(emergency) + ".zip",
            )
        finally:
            raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handler)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def find_dataset(project_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [
        project_root / "clean_dataset_2209.csv",
        project_root / "clean_dataset_2209(1).csv",
        project_root / "cluster_runs" / "large" / "summary_live.csv",
        project_root / "summary_live.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the dataset. Pass it explicitly with --dataset /path/to/file.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one dynamic general PPO model for MetaLore."
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--c2", type=float, default=None)
    parser.add_argument("--train-timesteps", type=int, default=DEFAULT_TRAIN_TIMESTEPS)
    parser.add_argument("--episode-steps", type=int, default=DEFAULT_EPISODE_STEPS)
    parser.add_argument("--load-hold-steps", type=int, default=DEFAULT_LOAD_HOLD_STEPS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--train-max-ues", type=int, default=TRAIN_MAX_UE)
    parser.add_argument("--train-max-sensors", type=int, default=TRAIN_MAX_SENSOR)
    parser.add_argument("--pool-max-ues", type=int, default=POOL_MAX_UE)
    parser.add_argument("--pool-max-sensors", type=int, default=POOL_MAX_SENSOR)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument(
        "--eval-episodes-per-target",
        type=int,
        default=DEFAULT_EVAL_EPISODES_PER_TARGET,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--start-method",
        choices=["auto", "spawn", "fork", "forkserver"],
        default="auto",
        help="Subprocess start method. auto uses spawn on Windows and fork on Linux.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="CPU threads used by the PPO neural-network updates.",
    )
    parser.add_argument("--output-dir", type=str, default="main_dynamic_ppo_big20")
    parser.add_argument(
        "--resume-auto",
        action="store_true",
        help="Resume from the newest checkpoint already present in the output directory.",
    )
    parser.add_argument(
        "--resume-model",
        type=str,
        default=None,
        help="Resume from a specific Stable-Baselines3 PPO .zip model.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not create a final tar.gz archive under results_backups/.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show the tqdm progress bar. Leave disabled for nohup cluster logs.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Small smoke test: 20k steps, 2 envs, 200-step episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    output_dir = (project_root / args.output_dir).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    monitor_dir = output_dir / "monitors"
    models_dir = output_dir / "models"
    for directory in (output_dir, checkpoints_dir, monitor_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    write_run_status(
        output_dir,
        "configuring",
        command=" ".join(sys.argv),
        oar_job_id=os.environ.get("OAR_JOB_ID"),
    )

    if args.quick_test:
        args.train_timesteps = 20_000
        args.episode_steps = 200
        args.load_hold_steps = 25
        args.num_envs = 2
        args.checkpoint_every = 10_000
        args.eval_episodes_per_target = 1

    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.load_hold_steps < 1 or args.load_hold_steps > args.episode_steps:
        raise ValueError("--load-hold-steps must be between 1 and --episode-steps")
    if args.train_max_ues > args.pool_max_ues:
        raise ValueError("--train-max-ues cannot exceed --pool-max-ues")
    if args.train_max_sensors > args.pool_max_sensors:
        raise ValueError("--train-max-sensors cannot exceed --pool-max-sensors")
    if args.train_max_ues < MIN_UE or args.train_max_sensors < MIN_SENSOR:
        raise ValueError("Training maxima must be at least the configured minima")

    dataset_path = find_dataset(project_root, args.dataset)
    print("Dataset:", dataset_path)

    eta, c2, _, reward_selection = choose_fixed_reward_pair(
        dataset_path=dataset_path,
        output_dir=output_dir,
        eta_override=args.eta,
        c2_override=args.c2,
    )
    capacity = estimate_capacity(seed=args.seed)

    empirical_max = reward_selection.get("empirical_max_supported_total_devices")
    theoretical_average = capacity["approx_total_device_limits"][
        "communication_average_position"
    ]
    compute_limit = capacity["approx_total_device_limits"]["computation"]
    if empirical_max is not None:
        support_total = int(min(empirical_max, theoretical_average, compute_limit))
    else:
        support_total = int(min(theoretical_average, compute_limit))

    with (output_dir / "reward_selection.json").open("w", encoding="utf-8") as f:
        json.dump(reward_selection, f, indent=2)
    with (output_dir / "capacity_report.json").open("w", encoding="utf-8") as f:
        json.dump(capacity, f, indent=2)
    create_capacity_boundary_plot(
        capacity=capacity,
        output_dir=output_dir,
        train_max_ues=args.train_max_ues,
        train_max_sensors=args.train_max_sensors,
        pool_max_ues=args.pool_max_ues,
        pool_max_sensors=args.pool_max_sensors,
    )

    print("\n=== FIXED REWARD PARAMETERS ===")
    print(f"eta = {eta}")
    print(f"C2  = {c2}")
    print("Selection source:", reward_selection["source"])
    print("Empirical supported total from dataset:", empirical_max)

    print("\n=== CAPACITY ESTIMATE ===")
    for key, value in capacity["approx_total_device_limits"].items():
        print(f"{key}: {value}")
    print("Practical support total used by training curriculum:", support_total)
    print(
        "WARNING: the requested 100 UE + 50 sensor state has 150 devices. "
        "It is an overload state under the current 600 MHz / 800-unit BS. "
        "The training includes some overload episodes so PPO learns graceful degradation, "
        "but 150 devices cannot be claimed as fully supported."
    )

    config = build_dynamic_config(
        eta=eta,
        c2=c2,
        episode_steps=args.episode_steps,
        load_hold_steps=args.load_hold_steps,
        seed=args.seed,
        support_total=support_total,
        train_max_ues=args.train_max_ues,
        train_max_sensors=args.train_max_sensors,
        pool_max_ues=args.pool_max_ues,
        pool_max_sensors=args.pool_max_sensors,
    )

    monitor_run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_fns = [
        make_env_factory(
            config,
            rank=rank,
            monitor_dir=monitor_dir,
            run_tag=monitor_run_tag,
        )
        for rank in range(args.num_envs)
    ]
    if args.num_envs > 1:
        if args.start_method == "auto":
            start_method = "spawn" if platform.system().lower().startswith("win") else "fork"
        else:
            start_method = args.start_method
        print("Parallel environment start method:", start_method)
        train_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        start_method = "none"
        train_env = DummyVecEnv(env_fns)

    torch.set_num_threads(max(1, int(args.torch_threads)))
    print("PyTorch CPU threads:", torch.get_num_threads())

    save_freq_calls = max(1, int(args.checkpoint_every // args.num_envs))
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=str(checkpoints_dir),
        name_prefix="main_dynamic_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Keep the total PPO rollout batch near 2,048 samples when num_envs changes.
    # Examples: 1 env -> 2048 steps, 4 envs -> 512 each, 8 envs -> 256 each,
    # 12 envs -> 256 each (3,072 total). This keeps learning behavior comparable
    # between the laptop and cluster commands instead of changing only speed.
    target_rollout_size = 2_048
    rollout_steps_per_env = max(
        128,
        int(math.ceil((target_rollout_size / args.num_envs) / 128.0) * 128),
    )
    rollout_size = rollout_steps_per_env * args.num_envs

    batch_size = 256 if rollout_size >= 256 else rollout_size
    while rollout_size % batch_size != 0 and batch_size > 16:
        batch_size //= 2

    resume_path: Optional[Path] = None
    if args.resume_model:
        resume_path = Path(args.resume_model).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
    elif args.resume_auto:
        resume_path = find_latest_resume_model(output_dir)

    if resume_path is not None:
        print("Resuming PPO from:", resume_path)
        model = PPO.load(str(resume_path), env=train_env, device=args.device)
        # Keep the current run's CPU/logging settings.
        model.tensorboard_log = str(output_dir / "tensorboard")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=rollout_steps_per_env,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.20,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
            device=args.device,
        )

    install_emergency_signal_handlers(model, output_dir)

    settings = {
        "fixed_eta": eta,
        "fixed_c2": c2,
        "sync_base_reward": SYNC_BASE_REWARD,
        "e2e_delay_threshold": E2E_DELAY_THRESHOLD,
        "min_resource_share": MIN_RESOURCE_SHARE,
        "training_ue_range": [MIN_UE, int(args.train_max_ues), UE_STEP],
        "training_sensor_range": [MIN_SENSOR, int(args.train_max_sensors), SENSOR_STEP],
        "stress_test_pool": [int(args.pool_max_ues), int(args.pool_max_sensors)],
        "practical_support_total": support_total,
        "train_timesteps_maximum": int(args.train_timesteps),
        "episode_steps": int(args.episode_steps),
        "load_hold_steps": int(args.load_hold_steps),
        "num_parallel_envs_one_policy": int(args.num_envs),
        "subprocess_start_method": start_method,
        "torch_threads": int(args.torch_threads),
        "phase_probabilities": {
            "supported_or_near_capacity": SUPPORTED_PHASE_PROBABILITY,
            "moderate_overload": MODERATE_PHASE_PROBABILITY,
            "strong_overload": STRONG_PHASE_PROBABILITY,
        },
        "checkpoint_every": int(args.checkpoint_every),
        "eval_episodes_per_target": int(args.eval_episodes_per_target),
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps_per_env": rollout_steps_per_env,
            "rollout_size": rollout_size,
            "batch_size": batch_size,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
            "ent_coef": 0.01,
            "network": [128, 128],
        },
    }
    with (output_dir / "training_settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print("\n=== TRAINING ONE GENERAL PPO MODEL ===")
    print("Maximum training timesteps:", f"{args.train_timesteps:,}")
    print("Episode length:", args.episode_steps)
    print("Parallel environments feeding the same PPO:", args.num_envs)
    print("Load changes every:", args.load_hold_steps, "steps")
    print("Training UE range:", f"{MIN_UE}..{args.train_max_ues} step {UE_STEP}")
    print("Training sensor range:", f"{MIN_SENSOR}..{args.train_max_sensors} step {SENSOR_STEP}")
    print("Approximate completed episodes across all envs:", args.train_timesteps // args.episode_steps)
    print("Output:", output_dir)

    starting_timesteps = int(model.num_timesteps)
    remaining_timesteps = max(0, int(args.train_timesteps) - starting_timesteps)
    print("Starting PPO timesteps:", f"{starting_timesteps:,}")
    print("Remaining PPO timesteps:", f"{remaining_timesteps:,}")

    write_run_status(
        output_dir,
        "training",
        starting_timesteps=starting_timesteps,
        target_timesteps=int(args.train_timesteps),
        remaining_timesteps=remaining_timesteps,
        eta=eta,
        c2=c2,
        num_envs=int(args.num_envs),
    )

    start = time.perf_counter()
    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=checkpoint_callback,
            progress_bar=bool(args.progress_bar),
            reset_num_timesteps=(starting_timesteps == 0),
        )
    else:
        print("Target timesteps already reached; skipping additional training.")
    elapsed = time.perf_counter() - start

    final_steps = int(model.num_timesteps)
    final_model = models_dir / f"main_dynamic_ppo_final_{final_steps}_steps.zip"
    model.save(str(final_model.with_suffix("")))
    train_env.close()

    # Make sure the exact final model participates in checkpoint selection.
    checkpoints = list(checkpoints_dir.glob("main_dynamic_ppo_*_steps.zip"))
    checkpoints.append(final_model)
    checkpoints = sorted(set(path.resolve() for path in checkpoints), key=checkpoint_timestep)

    print("\n=== SELECTING THE BEST TRAINING TIMESTEP ===")
    best_path, best_steps, validation_summary = select_best_checkpoint(
        checkpoints=checkpoints,
        output_dir=output_dir,
        eta=eta,
        c2=c2,
        episode_steps=args.episode_steps,
        load_hold_steps=args.load_hold_steps,
        support_total=support_total,
        train_max_ues=args.train_max_ues,
        train_max_sensors=args.train_max_sensors,
        pool_max_ues=args.pool_max_ues,
        pool_max_sensors=args.pool_max_sensors,
        eval_episodes_per_target=args.eval_episodes_per_target,
        seed=args.seed + 70_000,
        device=args.device,
    )

    selected_model = models_dir / "main_dynamic_ppo_best.zip"
    shutil.copy2(best_path, selected_model)

    final_report = {
        "training_seconds_this_invocation": elapsed,
        "starting_timesteps": starting_timesteps,
        "final_timesteps": final_steps,
        "fixed_eta": eta,
        "fixed_c2": c2,
        "practical_supported_total_devices": support_total,
        "training_maximum_state": {
            "num_ues": int(args.train_max_ues),
            "num_sensors": int(args.train_max_sensors),
        },
        "severe_stress_test_state": {
            "num_ues": int(args.pool_max_ues),
            "num_sensors": int(args.pool_max_sensors),
        },
        "best_checkpoint_source": str(best_path),
        "best_training_timesteps": best_steps,
        "selected_model": str(selected_model),
        "selection_rule": (
            "Choose the earliest checkpoint within 2% of the best weighted "
            "validation score, while preserving near-best supported-load "
            "completion and queue stability."
        ),
    }
    with (output_dir / "final_report.json").open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    create_result_plots(output_dir=output_dir, best_steps=best_steps)
    create_training_monitor_plots(output_dir=output_dir)
    evaluate_dynamic_response_trace(
        model_path=selected_model,
        output_dir=output_dir,
        eta=eta,
        c2=c2,
        episode_steps=args.episode_steps,
        load_hold_steps=args.load_hold_steps,
        support_total=support_total,
        train_max_ues=args.train_max_ues,
        train_max_sensors=args.train_max_sensors,
        pool_max_ues=args.pool_max_ues,
        pool_max_sensors=args.pool_max_sensors,
        seed=args.seed + 90_000,
        device=args.device,
    )

    archive: Optional[Path] = None
    if not args.no_archive:
        archive = create_results_archive(output_dir=output_dir, project_root=project_root)
        final_report["results_archive"] = str(archive)
        with (output_dir / "final_report.json").open("w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

    write_run_status(
        output_dir,
        "completed",
        final_timesteps=final_steps,
        best_training_timesteps=best_steps,
        selected_model=str(selected_model),
        results_archive=str(archive) if archive else None,
    )

    print("\n=== FINISHED ===")
    print("Training time this invocation (seconds):", round(elapsed, 2))
    print("Final PPO timesteps:", f"{final_steps:,}")
    print("Best training timestep:", f"{best_steps:,}")
    print("Best model:", selected_model)
    print("Validation summary:", output_dir / "checkpoint_validation_summary.csv")
    print("Dynamic response trace:", output_dir / "dynamic_response_trace.csv")
    print("Plots:", output_dir / "plots")
    print("Final report:", output_dir / "final_report.json")
    if archive is not None:
        print("Portable results archive:", archive)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        try:
            failed_args = parse_args()
            failed_output = (Path.cwd().resolve() / failed_args.output_dir).resolve()
            failed_output.mkdir(parents=True, exist_ok=True)
            write_run_status(
                failed_output,
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        except Exception:
            pass
        raise
