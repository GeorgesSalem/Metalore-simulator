"""
Smart City Handler for MetaLore Environments.

Defines the RL interface for smart city scenarios:
    - Action: [bandwidth_allocation, compute_allocation]
    - Observation: [ue_queue_length, sensor_queue_length]
    - Reward: Based on synchronization and delay penalties
"""

from typing import Dict, Tuple
import numpy as np
from gymnasium import spaces

from metalore.handlers.handler import Handler

class SmartCityHandler(Handler):

    @classmethod
    def action_space(cls, env) -> spaces.Space:
        """Define continuous action space for bandwidth and compute split."""
        min_share = env.config["reward"].get("min_resource_share", 0.20)

        low = np.array([min_share, min_share], dtype=np.float32)
        high = np.array([1.0 - min_share, 1.0 - min_share], dtype=np.float32)

        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    @classmethod
    def observation_space(cls, env) -> spaces.Space:
        low = np.zeros(8, dtype=np.float32)
        high = np.ones(8, dtype=np.float32) * 1e4
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Computes observations for agent."""

        ue_tx_q = sum(ue.tx_queue.length for ue in env.active_ues)
        sensor_tx_q = sum(s.tx_queue.length for s in env.active_sensors)

        ue_proc_q = sum(bs.proc_queues['UE'].length for bs in env.stations.values())
        sensor_proc_q = sum(bs.proc_queues['SENSOR'].length for bs in env.stations.values())

        n_ue = max(1, env.num_ues)
        n_sensor = max(1, env.num_sensors)

        latest_aori = env.metrics.latest("mean_aori") or 0.0
        latest_aosi = env.metrics.latest("mean_aosi") or 0.0

        return np.array([
            ue_tx_q / n_ue,
            sensor_tx_q / n_sensor,
            ue_proc_q / n_ue,
            sensor_proc_q / n_sensor,
            len(env.active_ues) / 100.0,
            len(env.active_sensors) / 100.0,
            latest_aori,
            latest_aosi,
        ], dtype=np.float32)

    @classmethod
    def action(cls, env, actions) -> Tuple[float, float]:
        min_share = env.config["reward"].get("min_resource_share", 0.05)

        return (
            float(np.clip(actions[0], min_share, 1.0 - min_share)),
            float(np.clip(actions[1], min_share, 1.0 - min_share)),
        )

    
    @classmethod
    def reward(cls, env) -> float:
        """Computes rewards for agent."""
        reward_cfg = env.config['reward']

        delay_threshold = reward_cfg['e2e_delay_threshold']
        delay_penalty = reward_cfg['delay_penalty']
        sync_base_reward = reward_cfg['sync_base_reward']
        discount_factor = reward_cfg['discount_factor']

        # UE jobs fully processed this timestep
        step_ue_jobs = [
            job for job in env.job_tracker.step_completed_jobs
            if job.entity_type == 'UE'
        ]

        reward = 0.0

        # Part 1: delay penalty
        # Applied once per UE job that exceeded the e2e delay threshold
        for job in step_ue_jobs:
            if job.aori is not None and job.aori > delay_threshold:
                reward += delay_penalty

        # Part 2: synchronization reward
        # Discounted by how stale the sensor data was
        for job in step_ue_jobs:
            if job.aosi is not None:
                reward += sync_base_reward * (discount_factor ** job.aosi)

        return reward

    
    @classmethod
    def check(cls, env) -> None:
        """Check if handler is applicable to simulation configuration."""
        pass
    
    @classmethod
    def info(cls, env) -> Dict:
        """Compute information for feedback loop."""
        ue_rates = {ue.id: rate for (_, ue), rate in env.datarates_ue.items()}
        sensor_rates = {s.id: rate for (_, s), rate in env.datarates_sensor.items()}

        return {
            'time': env.time,
            'num_bs': env.num_bs,
            'num_ues': env.num_ues,
            'num_sensors': env.num_sensors,
            'num_active_users': len(env.active_ues),
            'num_active_sensors': len(env.active_sensors),
            'ue_datarates': ue_rates,
            'sensor_datarates': sensor_rates,
        }
    