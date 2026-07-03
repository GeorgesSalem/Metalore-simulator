import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from make_env import make_env, describe_env
from run_config import (
    NEW_MODEL_PATH, RESULTS_DIR, CHECKPOINTS_DIR, SCENARIO, TRAIN_TIMESTEPS,
    LEARNING_RATE, N_STEPS, BATCH_SIZE, GAMMA, N_EPOCHS, SEED,
)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    NEW_MODEL_PATH.parent.mkdir(exist_ok=True)

    env = make_env(SCENARIO, seed=SEED)
    env = Monitor(env, filename=str(RESULTS_DIR / "monitor_train_new.csv"))

    print("Training a new PPO model from zero.")
    print("Scenario:", SCENARIO)
    print("Environment:", describe_env(env.unwrapped if hasattr(env, "unwrapped") else env))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        n_epochs=N_EPOCHS,
        seed=SEED,
        verbose=1,
        tensorboard_log=str(RESULTS_DIR / "tensorboard"),
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(CHECKPOINTS_DIR),
        name_prefix="ppo_new_checkpoint",
    )

    start = time.perf_counter()
    model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)
    elapsed = time.perf_counter() - start

    model.save(str(NEW_MODEL_PATH))

    print("\nTraining finished.")
    print("Training time:", round(elapsed, 2), "seconds")
    print("Speed:", round(TRAIN_TIMESTEPS / elapsed, 2), "steps/second")
    print("Saved model:", NEW_MODEL_PATH)

    env.close()


if __name__ == "__main__":
    main()
