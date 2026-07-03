import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from make_env import make_env, describe_env
from run_config import (
    MODEL_PATH, CONTINUED_MODEL_PATH, RESULTS_DIR, CHECKPOINTS_DIR,
    SCENARIO, TRAIN_TIMESTEPS, SEED,
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    env = make_env(SCENARIO, seed=SEED)
    env = Monitor(env, filename=str(RESULTS_DIR / "monitor_continue_training.csv"))

    model = PPO.load(str(MODEL_PATH), env=env, device="cpu", tensorboard_log=str(RESULTS_DIR / "tensorboard"))

    print("Continuing training from:", MODEL_PATH)
    print("Scenario:", SCENARIO)
    print("Environment:", describe_env(env.unwrapped if hasattr(env, "unwrapped") else env))
    print("Extra timesteps:", TRAIN_TIMESTEPS)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(CHECKPOINTS_DIR),
        name_prefix="ppo_continue_checkpoint",
    )

    start = time.perf_counter()
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        reset_num_timesteps=False,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - start

    model.save(str(CONTINUED_MODEL_PATH))

    print("\nTraining finished.")
    print("Training time:", round(elapsed, 2), "seconds")
    print("Speed:", round(TRAIN_TIMESTEPS / elapsed, 2), "steps/second")
    print("Saved new model:", CONTINUED_MODEL_PATH)

    env.close()


if __name__ == "__main__":
    main()
