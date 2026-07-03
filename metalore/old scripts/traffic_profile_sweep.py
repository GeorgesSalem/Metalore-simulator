from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

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

from make_env import make_env_from_config
from metalore.config import dynamic_traffic_config
from metalore.scenarios import SingleCellEnv


ALL_PROFILES = ["low", "medium", "high", "ramp_up", "ramp_down", "burst"]


def build_traffic_config(profile: str, seed: int):
    config = dynamic_traffic_config(profile)

    config["environment"]["seed"] = seed

    return config


def evaluate_profile(model: PPO, profile: str, episodes: int, seed: int):
    config = build_traffic_config(profile, seed)
    env = make_env_from_config(config, env_cls=SingleCellEnv, seed=seed)

    episode_rows = []
    step_rows = []
    all_job_rows = []

    for episode in range(1, episodes + 1):
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
                "profile": profile,
                "episode": episode,
                "step": step,
                "reward": float(reward),
                "total_reward_so_far": total_reward,
                "action_bw_split": float(action[0]),
                "action_comp_split": float(action[1]),
            }
            row.update(latest)
            step_rows.append(row)
            obs = next_obs

        ep_summary = env.metrics.finalize(env.job_tracker)
        job_df = env.job_tracker.to_dataframe()
        if not job_df.empty:
            job_df.insert(0, "profile", profile)
            job_df.insert(1, "episode", episode)
            all_job_rows.append(job_df)

        episode_rows.append({
            "profile": profile,
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
    jobs = pd.concat(all_job_rows, ignore_index=True) if all_job_rows else pd.DataFrame()
    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows), jobs


def summarize(episodes: pd.DataFrame, steps: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    summary = episodes.groupby("profile", as_index=False).agg({
        "steps": "mean",
        "total_reward": "mean",
        "jobs_generated": "mean",
        "jobs_transmitted": "mean",
        "jobs_processed": "mean",
        "job_completion_rate": "mean",
        "mean_aori": "mean",
        "mean_aosi": "mean",
    })

    actions = steps.groupby("profile", as_index=False).agg({
        "num_active_ues": "mean",
        "num_active_sensors": "mean",
        "action_bw_split": "mean",
        "action_comp_split": "mean",
    })
    summary = summary.merge(actions, on="profile", how="left")

    if not jobs.empty and "aori" in jobs.columns and "aosi" in jobs.columns:
        ue_jobs = jobs[jobs.get("entity_type") == "UE"].copy()
        if not ue_jobs.empty:
            p95 = ue_jobs.groupby("profile", as_index=False).agg(
                p95_aori=("aori", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95)),
                p95_aosi=("aosi", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95)),
            )
            summary = summary.merge(p95, on="profile", how="left")

    order = {p: i for i, p in enumerate(ALL_PROFILES)}
    summary["profile_order"] = summary["profile"].map(order)
    return summary.sort_values("profile_order").drop(columns=["profile_order"]).reset_index(drop=True)


def save_plots(summary: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    x = range(len(summary))
    labels = summary["profile"].tolist()

    def setup(ax, ylabel, title):
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Traffic profile")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(list(x), summary["mean_aori"], marker="o", label="Mean AoRI")
    ax.plot(list(x), summary["mean_aosi"], marker="o", label="Mean AoSI")
    setup(ax, "Mean information age (steps)", "AoRI and AoSI by Dynamic Traffic Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "aori_aosi_by_profile.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(list(x), summary["job_completion_rate"] * 100.0, marker="o")
    setup(ax, "Completion rate (%)", "Completion Rate by Dynamic Traffic Profile")
    fig.tight_layout()
    fig.savefig(output_dir / "completion_rate_by_profile.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(list(x), summary["jobs_generated"], marker="o", label="Generated")
    ax.plot(list(x), summary["jobs_transmitted"], marker="o", label="Transmitted")
    ax.plot(list(x), summary["jobs_processed"], marker="o", label="Processed")
    setup(ax, "Jobs per episode", "Jobs Flow by Dynamic Traffic Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jobs_flow_by_profile.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(list(x), summary["action_bw_split"], marker="o", label="Pp bandwidth to UEs")
    ax.plot(list(x), summary["action_comp_split"], marker="o", label="Pc compute to UEs")
    setup(ax, "UE resource share", "Resource Allocation by Dynamic Traffic Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "resource_allocation_by_profile.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(list(x), summary["num_active_ues"], marker="o", label="Active UEs")
    ax.plot(list(x), summary["num_active_sensors"], marker="o", label="Active sensors")
    setup(ax, "Average active devices", "Average Active Devices by Dynamic Traffic Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "active_devices_by_profile.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    # You can still change these from terminal, but they have default values.
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5555)
    parser.add_argument("--profiles", nargs="+", default=ALL_PROFILES, choices=ALL_PROFILES)

    args = parser.parse_args()


    EXPERIMENT_NAME = "mixed_Profiles_rewardA_300k"

    model_path = METALORE_DIR / "models" / f"ppo_traffic_{EXPERIMENT_NAME}.zip"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = (
        METALORE_DIR
        / "results"
        / "dynamic_traffic"
        / "profile_sweep"
        / f"eval_{EXPERIMENT_NAME}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating PPO traffic-aware model")
    print("Experiment:", EXPERIMENT_NAME)
    print("Using model:", model_path)
    print("Profiles:", ", ".join(args.profiles))
    print("Episodes per profile:", args.episodes)
    print("Seed:", args.seed)
    print("Output dir:", output_dir)

    model = PPO.load(str(model_path), device="cpu")

    all_episodes = []
    all_steps = []
    all_jobs = []

    start = time.perf_counter()

    for profile in args.profiles:
        print(f"\nEvaluating profile: {profile}")

        episodes, steps, jobs = evaluate_profile(
            model=model,
            profile=profile,
            episodes=args.episodes,
            seed=args.seed,
        )

        all_episodes.append(episodes)
        all_steps.append(steps)

        if not jobs.empty:
            all_jobs.append(jobs)

        print("Completion:", round(episodes["job_completion_rate"].mean() * 100.0, 1), "%")
        print("AoRI:", round(episodes["mean_aori"].mean(), 3))
        print("AoSI:", round(episodes["mean_aosi"].mean(), 3))

    episodes_df = pd.concat(all_episodes, ignore_index=True)
    steps_df = pd.concat(all_steps, ignore_index=True)
    jobs_df = pd.concat(all_jobs, ignore_index=True) if all_jobs else pd.DataFrame()

    summary = summarize(episodes_df, steps_df, jobs_df)

    episodes_df.to_csv(output_dir / "profile_episodes.csv", index=False)
    steps_df.to_csv(output_dir / "profile_steps.csv", index=False)

    if not jobs_df.empty:
        jobs_df.to_csv(output_dir / "profile_jobs.csv", index=False)

    summary.to_csv(output_dir / "summary.csv", index=False)
    save_plots(summary, output_dir)

    print("\n=== DYNAMIC TRAFFIC PROFILE SUMMARY ===")

    cols = [
        "profile",
        "job_completion_rate",
        "mean_aori",
        "mean_aosi",
        "p95_aori",
        "p95_aosi",
        "action_bw_split",
        "action_comp_split",
        "total_reward",
    ]

    cols = [c for c in cols if c in summary.columns]

    print(summary[cols].round(3).to_string(index=False))

    elapsed = time.perf_counter() - start

    print("\nFinished in", round(elapsed, 2), "seconds")
    print("Saved summary:", output_dir / "summary.csv")
    print("Saved plots in:", output_dir)


if __name__ == "__main__":
    main()
