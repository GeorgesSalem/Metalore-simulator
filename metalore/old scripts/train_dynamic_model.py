from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from make_env import describe_env, make_env_from_config
from metalore.config import (
    DYNAMIC_DEFAULTS,
    DYNAMIC_TRAINING,
    dynamic_model_filename,
    dynamic_training_config,
    reward_profile_label,
)
from metalore.scenarios import SingleCellEnv


def main():
    reward_profile = DYNAMIC_DEFAULTS["reward_profile"]
    train_timesteps = int(DYNAMIC_DEFAULTS["train_timesteps"])
    seed = int(DYNAMIC_DEFAULTS["seed"])

    models_dir = ROOT / "models"
    results_dir = ROOT / "results" / "dynamic_training" / f"{reward_profile}_{DYNAMIC_TRAINING['num_ues']}ue_{DYNAMIC_TRAINING['num_sensors']}s"
    checkpoints_dir = ROOT / "checkpoints" / "dynamic_training" / f"{reward_profile}_{DYNAMIC_TRAINING['num_ues']}ue_{DYNAMIC_TRAINING['num_sensors']}s"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / dynamic_model_filename(reward_profile, train_timesteps)

    config = dynamic_training_config(reward_profile)
    env = make_env_from_config(config, env_cls=SingleCellEnv, seed=seed)
    env = Monitor(env, filename=str(results_dir / "monitor_train_dynamic.csv"))

    print("Training PPO model from zero")
    print("Training config:", f"{DYNAMIC_TRAINING['num_ues']} UE / {DYNAMIC_TRAINING['num_sensors']} sensors")
    print("Timesteps:", train_timesteps)
    print("Reward profile:", reward_profile)
    print(reward_profile_label(reward_profile))
    print("Environment:", describe_env(env.unwrapped if hasattr(env, "unwrapped") else env))
    print("Model will be saved to:", model_path)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(DYNAMIC_DEFAULTS.get("learning_rate", 3e-4)),
        n_steps=int(DYNAMIC_DEFAULTS.get("n_steps", 2048)),
        batch_size=int(DYNAMIC_DEFAULTS.get("batch_size", 64)),
        gamma=float(DYNAMIC_DEFAULTS.get("gamma", 0.99)),
        gae_lambda=float(DYNAMIC_DEFAULTS.get("gae_lambda", 0.95)),
        clip_range=float(DYNAMIC_DEFAULTS.get("clip_range", 0.20)),
        ent_coef=float(DYNAMIC_DEFAULTS.get("ent_coef", 0.0)),
        vf_coef=float(DYNAMIC_DEFAULTS.get("vf_coef", 0.5)),
        max_grad_norm=float(DYNAMIC_DEFAULTS.get("max_grad_norm", 0.5)),
        n_epochs=int(DYNAMIC_DEFAULTS.get("n_epochs", 10)),
        seed=seed,
        verbose=1,
        tensorboard_log=str(results_dir / "tensorboard"),
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(checkpoints_dir),
        name_prefix="ppo_dynamic_checkpoint",
    )

    start = time.perf_counter()
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback, progress_bar=True)
    elapsed = time.perf_counter() - start

    model.save(str(model_path))
    env.close()

    print("\nTraining finished.")
    print("Training time:", round(elapsed, 2), "seconds")
    print("Speed:", round(train_timesteps / elapsed, 2), "steps/second")
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
