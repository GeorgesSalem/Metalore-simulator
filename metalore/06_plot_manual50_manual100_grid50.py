from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from static_reward_common import METALORE_DIR


# ============================================================
# INPUT RESULT FOLDERS
# Change these names only if your folders are different
# ============================================================

BASE_DIR = METALORE_DIR / "results" / "reward_pipeline" / "static_25s"

EXPERIMENTS = {
    "Manual 50k": BASE_DIR
    / "manual_static_25s_eta0p7_c2m2p0_user_priority_50k"
    / "manual_best_by_ue.csv",

    "Manual 100k": BASE_DIR
    / "manual_static_25s_eta0p7_c2m2p0_user_priority_100k"
    / "manual_best_by_ue.csv",

    "Grid best 50k": BASE_DIR
    / "grid_static_25s_user_priority_50k"
    / "grid_best_by_ue.csv",
}

OUTPUT_DIR = BASE_DIR / "plots_manual50_manual100_grid50"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    frames = []

    for label, path in EXPERIMENTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing result file for {label}: {path}")

        df = pd.read_csv(path)
        df["method"] = label
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["num_ues", "method"]).reset_index(drop=True)

    return all_df


def ppo_label(num_ues: int, num_sensors: int) -> str:
    mapping = {
        10: "PPO1\n10 UE / 25 S",
        20: "PPO2\n20 UE / 25 S",
        30: "PPO3\n30 UE / 25 S",
        40: "PPO4\n40 UE / 25 S",
        50: "PPO5\n50 UE / 25 S",
    }
    return mapping.get(int(num_ues), f"{int(num_ues)} UE / {int(num_sensors)} S")


def plot_grouped_metric(df, metric, ylabel, title, filename):
    required = ["num_ues", "num_sensors", "method", metric]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Skipping {metric}. Missing columns: {missing}")
        return

    pivot = df.pivot_table(
        index=["num_ues", "num_sensors"],
        columns="method",
        values=metric,
        aggfunc="mean",
    )

    pivot = pivot.sort_index()

    labels = [
        ppo_label(num_ues, num_sensors)
        for num_ues, num_sensors in pivot.index
    ]

    methods = list(EXPERIMENTS.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(methods)

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        if method not in pivot.columns:
            continue

        values = pivot[method].values
        positions = x - 0.4 + width / 2 + i * width
        plt.bar(positions, values, width, label=method)

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_jobs_flow(df):
    needed = [
        "num_ues",
        "num_sensors",
        "method",
        "jobs_generated",
        "jobs_transmitted",
        "jobs_processed",
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Skipping jobs flow. Missing columns: {missing}")
        return

    for method in EXPERIMENTS.keys():
        sub = df[df["method"] == method].sort_values("num_ues")

        if sub.empty:
            continue

        labels = [
            ppo_label(row["num_ues"], row["num_sensors"])
            for _, row in sub.iterrows()
        ]

        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(12, 6))

        plt.bar(x - width, sub["jobs_generated"], width, label="Generated")
        plt.bar(x, sub["jobs_transmitted"], width, label="Transmitted")
        plt.bar(x + width, sub["jobs_processed"], width, label="Processed")

        plt.xticks(x, labels)
        plt.ylabel("Jobs per episode")
        plt.title(f"Jobs flow — {method}")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        safe_method = method.lower().replace(" ", "_")
        out_path = OUTPUT_DIR / f"jobs_flow_{safe_method}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print("Saved:", out_path)


def save_combined_csv(df):
    out_path = OUTPUT_DIR / "manual50_manual100_grid50_combined.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)


def main():
    df = load_results()

    print("\n=== Loaded results ===")
    print(df[[
        "method",
        "num_ues",
        "num_sensors",
        "eta",
        "c2",
        "mean_aori",
        "mean_aosi",
        "job_completion_rate",
        "throughput_jobs_per_step",
        "training_time_seconds",
        "reward_convergence_timestep",
    ]].round(4).to_string(index=False))

    save_combined_csv(df)

    plot_grouped_metric(
        df,
        metric="mean_aori",
        ylabel="Mean AoRI",
        title="Mean AoRI comparison",
        filename="aori_manual50_manual100_grid50.png",
    )

    plot_grouped_metric(
        df,
        metric="mean_aosi",
        ylabel="Mean AoSI",
        title="Mean AoSI comparison",
        filename="aosi_manual50_manual100_grid50.png",
    )

    plot_grouped_metric(
        df,
        metric="throughput_jobs_per_step",
        ylabel="Processed jobs per timestep",
        title="Throughput comparison",
        filename="throughput_manual50_manual100_grid50.png",
    )

    plot_grouped_metric(
        df,
        metric="job_completion_rate",
        ylabel="Completion rate",
        title="Completion rate comparison",
        filename="completion_manual50_manual100_grid50.png",
    )

    plot_grouped_metric(
        df,
        metric="training_time_seconds",
        ylabel="Training time seconds",
        title="Training time comparison",
        filename="training_time_manual50_manual100_grid50.png",
    )

    plot_grouped_metric(
        df,
        metric="reward_convergence_timestep",
        ylabel="Convergence timestep",
        title="Reward convergence timestep comparison",
        filename="convergence_manual50_manual100_grid50.png",
    )

    plot_jobs_flow(df)

    print("\nAll plots saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()