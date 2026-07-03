from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from make_env import make_env_from_config
from metalore.config import (
    DYNAMIC_DEFAULTS,
    DYNAMIC_REWARD_PROFILES,
    DYNAMIC_SWEEP,
    DYNAMIC_TRAINING,
    dynamic_eval_config,
    dynamic_model_filename,
    dynamic_sweep_points,
    reward_profile_label,
)
from metalore.scenarios import SingleCellEnv


def evaluate_one_point(model: PPO, point: dict, reward_profile: str, eval_episodes: int, seed: int):
    config = dynamic_eval_config(point["num_ues"], point["num_sensors"], reward_profile=reward_profile)
    env = make_env_from_config(config, env_cls=SingleCellEnv, seed=seed)

    episode_rows = []
    step_rows = []

    for episode in range(1, eval_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += float(reward)
            step += 1

            latest = {
                k: v[-1]
                for k, v in env.metrics.step_totals.items()
                if v and k != "observation"
            }

            row = {
                "sweep_name": DYNAMIC_SWEEP["name"],
                "mode": point["mode"],
                "scenario_name": point["scenario_name"],
                "index": point["index"],
                "num_ues": point["num_ues"],
                "num_sensors": point["num_sensors"],
                "total_devices": point["total_devices"],
                "x_label": point["x_label"],
                "episode": episode,
                "step": step,
                "reward": float(reward),
                "action_bw_split": float(action[0]),
                "action_comp_split": float(action[1]),
            }
            row.update(latest)
            step_rows.append(row)
            obs = next_obs

        ep_summary = env.metrics.finalize(env.job_tracker)
        episode_rows.append({
            "sweep_name": DYNAMIC_SWEEP["name"],
            "mode": point["mode"],
            "scenario_name": point["scenario_name"],
            "index": point["index"],
            "num_ues": point["num_ues"],
            "num_sensors": point["num_sensors"],
            "total_devices": point["total_devices"],
            "target_split": point["target_split"],
            "x_label": point["x_label"],
            "episode": episode,
            "steps": step,
            "total_reward": total_reward,
            "jobs_generated": ep_summary.get("jobs_generated"),
            "jobs_transmitted": ep_summary.get("jobs_transmitted"),
            "jobs_processed": ep_summary.get("jobs_processed"),
            "job_completion_rate": ep_summary.get("job_completion_rate"),
            "mean_aori": ep_summary.get("mean_aori"),
            "mean_aosi": ep_summary.get("mean_aosi"),
        })

    env.close()
    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows)


def summarize_results(episodes: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    summary = episodes.groupby(
        ["sweep_name", "mode", "scenario_name", "index", "num_ues", "num_sensors", "total_devices", "target_split", "x_label"],
        as_index=False,
    ).agg({
        "jobs_generated": "mean",
        "jobs_transmitted": "mean",
        "jobs_processed": "mean",
        "job_completion_rate": "mean",
        "mean_aori": "mean",
        "mean_aosi": "mean",
        "total_reward": "mean",
    })

    actions = steps.groupby(["scenario_name"], as_index=False).agg({
        "action_bw_split": "mean",
        "action_comp_split": "mean",
    })

    summary = summary.merge(actions, on="scenario_name", how="left")
    return summary.sort_values("index").reset_index(drop=True)


def _setup_axis(ax, summary: pd.DataFrame, ylabel: str, title: str, reward_text: str):
    x = summary["index"].tolist()
    labels = summary["x_label"].tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Evaluation load")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{reward_text}")
    ax.grid(True)


def save_plots(summary: pd.DataFrame, output_dir: Path, reward_profile: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_text = reward_profile_label(reward_profile)
    x = summary["index"]

    # Plot 1: AoRI / AoSI
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, summary["mean_aori"], marker="o", label="Mean AoRI")
    ax.plot(x, summary["mean_aosi"], marker="o", label="Mean AoSI")
    _setup_axis(ax, summary, "Mean information age (timesteps)", "AoRI and AoSI vs Evaluation Load", reward_text)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "aori_aosi_vs_load.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Plot 2: Pp / Pc resource split
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, summary["target_split"], marker="o", label="Target UE ratio")
    ax.plot(x, summary["action_bw_split"], marker="o", label="Pp bandwidth to UEs")
    ax.plot(x, summary["action_comp_split"], marker="o", label="Pc compute to UEs")
    _setup_axis(ax, summary, "UE resource share", "Resource Allocation vs Evaluation Load", reward_text)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "resource_allocation_vs_load.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Plot 3: Completion rate
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, summary["job_completion_rate"] * 100.0, marker="o")
    _setup_axis(ax, summary, "Completion rate (%)", "Completion Rate vs Evaluation Load", reward_text)
    fig.tight_layout()
    fig.savefig(output_dir / "completion_rate_vs_load.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Plot 4: Jobs flow
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, summary["jobs_generated"], marker="o", label="Generated")
    ax.plot(x, summary["jobs_transmitted"], marker="o", label="Transmitted")
    ax.plot(x, summary["jobs_processed"], marker="o", label="Processed")
    _setup_axis(ax, summary, "Jobs per episode", "Jobs Flow vs Evaluation Load", reward_text)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jobs_flow_vs_load.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Extra useful plot: mean episode reward
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, summary["total_reward"], marker="o")
    _setup_axis(ax, summary, "Mean episode reward", "Reward vs Evaluation Load", reward_text)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_vs_load.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    reward_profile = DYNAMIC_DEFAULTS["reward_profile"]
    train_timesteps = int(DYNAMIC_DEFAULTS["train_timesteps"])
    eval_episodes = int(DYNAMIC_DEFAULTS["eval_episodes"])
    seed = int(DYNAMIC_DEFAULTS["seed"])

    model_path = ROOT / "models" / dynamic_model_filename(reward_profile, train_timesteps)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"First run: python train_dynamic_model.py\n"
            f"Current training point in default.py: {DYNAMIC_TRAINING['num_ues']} UE / {DYNAMIC_TRAINING['num_sensors']} sensors\n"
            f"Current reward_profile in default.py: {reward_profile}"
        )

    points = dynamic_sweep_points()
    if not points:
        raise ValueError("No sweep points were generated. Check DYNAMIC_SWEEP in config/default.py")

    output_dir = (
        ROOT / "results" / "dynamic_sweeps" / DYNAMIC_SWEEP["name"] /
        f"{reward_profile}_model_{DYNAMIC_TRAINING['num_ues']}ue_{DYNAMIC_TRAINING['num_sensors']}s"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using model:", model_path)
    print("Sweep:", DYNAMIC_SWEEP["name"], "mode=", DYNAMIC_SWEEP["mode"])
    print("Reward profile:", reward_profile)
    print(reward_profile_label(reward_profile))
    print("Evaluation episodes per point:", eval_episodes)
    print("Output dir:", output_dir)

    model = PPO.load(str(model_path), device="cpu")

    all_episodes = []
    all_steps = []
    start = time.perf_counter()

    for point in points:
        print(f"\nEvaluating {point['scenario_name']}: {point['num_ues']} UE / {point['num_sensors']} sensors")
        episodes_df, steps_df = evaluate_one_point(
            model=model,
            point=point,
            reward_profile=reward_profile,
            eval_episodes=eval_episodes,
            seed=seed,
        )
        all_episodes.append(episodes_df)
        all_steps.append(steps_df)
        print("Completion:", round(episodes_df["job_completion_rate"].mean() * 100, 1), "%")
        print("AoRI:", round(episodes_df["mean_aori"].mean(), 3))
        print("AoSI:", round(episodes_df["mean_aosi"].mean(), 3))

    episodes = pd.concat(all_episodes, ignore_index=True)
    steps = pd.concat(all_steps, ignore_index=True)
    summary = summarize_results(episodes, steps)

    episodes_path = output_dir / "episodes.csv"
    steps_path = output_dir / "steps.csv"
    summary_path = output_dir / "summary.csv"

    episodes.to_csv(episodes_path, index=False)
    steps.to_csv(steps_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_plots(summary, output_dir, reward_profile)

    print("\n=== DYNAMIC SWEEP SUMMARY ===")
    cols = [
        "num_ues", "num_sensors", "total_devices", "job_completion_rate",
        "mean_aori", "mean_aosi", "action_bw_split", "action_comp_split", "total_reward"
    ]
    print(summary[cols].round(3).to_string(index=False))

    elapsed = time.perf_counter() - start
    print("\nFinished in", round(elapsed, 2), "seconds")
    print("Saved summary:", summary_path)
    print("Saved plots in:", output_dir)


if __name__ == "__main__":
    main()
