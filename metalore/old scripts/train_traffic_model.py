from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parent

# This script can be placed either inside metalore/ or in metalore/scripts/.
# Find the folder that contains make_env.py.
if (ROOT / "make_env.py").exists():
    METALORE_DIR = ROOT
else:
    METALORE_DIR = ROOT.parent

PROJECT_ROOT = METALORE_DIR.parent

if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(METALORE_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from make_env import make_env_from_config
from metalore.config import dynamic_traffic_config
from metalore.scenarios import SingleCellEnv




EXPERIMENT_NAME = "mixed_Profiles_rewardA_300k"

TOTAL_TIMESTEPS = 300_000
SEED = 5555

# Profiles used during mixed training.
# The model will randomly train on these scenarios.
MIXED_PROFILES = ["high", "burst", "ramp_up", "ramp_down"]

# Probability of choosing each profile at the beginning of an episode.
# Same order as MIXED_PROFILES.
PROFILE_WEIGHTS = [0.40, 0.25, 0.20, 0.15]


def build_traffic_config(profile: str, seed: int):
    config = dynamic_traffic_config(profile)

    # Reward parameters are defined in:
    # metalore/config/default.py
    config["environment"]["seed"] = seed

    return config


class MixedTrafficEnv(gym.Env):
    """
    Gym wrapper that chooses a different traffic profile at each episode reset.

    Example:
    Episode 1 -> high
    Episode 2 -> burst
    Episode 3 -> ramp_up
    Episode 4 -> ramp_down
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        profiles: list[str],
        weights: list[float],
        seed: int,
    ):
        super().__init__()

        if len(profiles) != len(weights):
            raise ValueError("profiles and weights must have the same length.")

        self.profiles = profiles
        self.weights = weights
        self.base_seed = seed
        self.episode_id = 0
        self.current_profile = None
        self.current_env = None

        # Create one temporary environment only to get action/observation spaces.
        sample_profile = self.profiles[0]
        sample_config = build_traffic_config(sample_profile, self.base_seed)
        sample_env = make_env_from_config(
            sample_config,
            env_cls=SingleCellEnv,
            seed=self.base_seed,
        )

        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space

        sample_env.close()

    def _make_env_for_profile(self, profile: str):
        env_seed = self.base_seed + self.episode_id
        config = build_traffic_config(profile, env_seed)

        env = make_env_from_config(
            config,
            env_cls=SingleCellEnv,
            seed=env_seed,
        )

        return env

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.base_seed = seed

        self.episode_id += 1

        self.current_profile = random.choices(
            self.profiles,
            weights=self.weights,
            k=1,
        )[0]

        if self.current_env is not None:
            self.current_env.close()

        self.current_env = self._make_env_for_profile(self.current_profile)

        obs, info = self.current_env.reset(seed=self.base_seed + self.episode_id)

        if info is None:
            info = {}

        info["traffic_profile"] = self.current_profile
        info["episode_id"] = self.episode_id

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)

        if info is None:
            info = {}

        info["traffic_profile"] = self.current_profile
        info["episode_id"] = self.episode_id

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.current_env is not None:
            self.current_env.close()
            self.current_env = None


def main():
    models_dir = METALORE_DIR / "models"

    results_dir = (
        METALORE_DIR
        / "results"
        / "dynamic_traffic"
        / "training"
        / EXPERIMENT_NAME
    )

    checkpoints_dir = (
        METALORE_DIR
        / "checkpoints"
        / "dynamic_traffic"
        / EXPERIMENT_NAME
    )

    model_path = models_dir / f"ppo_traffic_{EXPERIMENT_NAME}.zip"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MixedTrafficEnv(
        profiles=MIXED_PROFILES,
        weights=PROFILE_WEIGHTS,
        seed=SEED,
    )

    env = Monitor(
        env,
        filename=str(results_dir / "monitor_train_traffic.csv"),
    )

    print("Training PPO traffic-aware model from zero")
    print("Experiment:", EXPERIMENT_NAME)
    print("Training mode: mixed dynamic traffic profiles")
    print("Profiles:", MIXED_PROFILES)
    print("Profile weights:", PROFILE_WEIGHTS)
    print("Timesteps:", TOTAL_TIMESTEPS)
    print("Seed:", SEED)
    print("Model will be saved to:", model_path)
    print("Results will be saved to:", results_dir)
    print("Checkpoints will be saved to:", checkpoints_dir)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        seed=SEED,
        verbose=1,
        tensorboard_log=str(results_dir / "tensorboard"),
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(checkpoints_dir),
        name_prefix=f"ppo_traffic_{EXPERIMENT_NAME}_checkpoint",
    )

    start = time.perf_counter()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    elapsed = time.perf_counter() - start

    model.save(str(model_path))
    env.close()

    print("\nTraining finished.")
    print("Training time:", round(elapsed, 2), "seconds")
    print("Speed:", round(TOTAL_TIMESTEPS / elapsed, 2), "steps/second")
    print("Saved model:", model_path)
    print("Saved training monitor:", results_dir / "monitor_train_traffic.csv")


if __name__ == "__main__":
    main()