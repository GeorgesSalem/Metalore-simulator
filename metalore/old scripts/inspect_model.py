from stable_baselines3 import PPO

from run_config import MODEL_PATH


def show_value(name, value):
    try:
        if callable(value):
            value = value(1.0)
    except Exception:
        pass
    print(f"{name}: {value}")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = PPO.load(str(MODEL_PATH), device="cpu", print_system_info=True)

    print("\nModel loaded successfully.")
    print("\nObservation space:", model.observation_space)
    print("Action space:", model.action_space)

    print("\nPolicy architecture:")
    print(model.policy)

    print("\nPPO hyperparameters:")
    for name in [
        "learning_rate", "gamma", "n_steps", "batch_size", "n_epochs",
        "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "clip_range",
        "num_timesteps", "seed", "n_envs"
    ]:
        if hasattr(model, name):
            show_value(name, getattr(model, name))

    print("\nNeural network parameter shapes:")
    for name, param in model.policy.named_parameters():
        print(name, tuple(param.shape))


if __name__ == "__main__":
    main()
