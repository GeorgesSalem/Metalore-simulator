from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from static_reward_common import (
    METALORE_DIR,
    NUM_SENSORS,
    OPERATOR_INTENTION,
    build_experiment_name,
)

# ============================================================
# STEP 5: plots
# This compares:
# 1. manual chosen reward
# 2. grid-search best reward per UE
# 3. LLM-candidate best reward per UE
# ============================================================

MANUAL_ETA = 0.8
MANUAL_C2 = -2.0

MANUAL_EXPERIMENT = "manual_static_25s_eta0p8_c2m2p0_user_priority_100k"
GRID_EXPERIMENT = "grid_ue10_30_50_70_90_static_25s_user_priority_100k"
LLM_EXPERIMENT = "llm_static_25s_user_priority_100k"

BASE_DIR = METALORE_DIR / "results" / "reward_pipeline" / "static_25s"
PLOT_DIR = BASE_DIR / f"plots_{OPERATOR_INTENTION}_manual_vs_grid_vs_llm"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_best(method: str, experiment: str) -> pd.DataFrame:
    if method == "grid":
        filename = "grid_ue10_30_50_70_90_best_by_ue.csv"
    else:
        filename = f"{method}_best_by_ue.csv"

    path = BASE_DIR / experiment / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    df = pd.read_csv(path)
    df["method_label"] = {
        "manual": "Manual reward",
        "grid": "Grid best",
        "llm": "LLM best",
    }.get(method, method)
    return df


def add_ppo_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    order = sorted(df["num_ues"].unique())
    mapping = {ue: f"PPO{i + 1}\n{ue} UE / {NUM_SENSORS} S" for i, ue in enumerate(order)}
    df["ppo_label"] = df["num_ues"].map(mapping)
    return df


def grouped_bar(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str):
    pivot = df.pivot(index="ppo_label", columns="method_label", values=metric)
    pivot = pivot.sort_index(key=lambda idx: [int(x.split("\n")[0].replace("PPO", "")) for x in idx])

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_title(title)
    ax.set_xlabel("Static PPO scenario")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def jobs_grouped(df: pd.DataFrame, method_label: str, filename: str):
    sub = df[df["method_label"] == method_label].copy()
    sub = add_ppo_labels(sub)
    pivot = sub.set_index("ppo_label")[["jobs_generated", "jobs_transmitted", "jobs_processed"]]
    pivot = pivot.sort_index(key=lambda idx: [int(x.split("\n")[0].replace("PPO", "")) for x in idx])

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_title(f"Jobs flow — {method_label}")
    ax.set_xlabel("Static PPO scenario")
    ax.set_ylabel("Average jobs per episode")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Metric")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def main():
    manual = load_best("manual", MANUAL_EXPERIMENT)
    grid = load_best("grid", GRID_EXPERIMENT)
    llm = load_best("llm", LLM_EXPERIMENT)

    df = pd.concat([manual, grid, llm], ignore_index=True)
    df = add_ppo_labels(df)
    df.to_csv(PLOT_DIR / "combined_best_results.csv", index=False)

    grouped_bar(
        df,
        metric="mean_aori",
        ylabel="AoRI in timesteps",
        title="Mean AoRI by static PPO scenario",
        filename="aori_manual_grid_llm.png",
    )
    grouped_bar(
        df,
        metric="mean_aosi",
        ylabel="AoSI in timesteps",
        title="Mean AoSI by static PPO scenario",
        filename="aosi_manual_grid_llm.png",
    )
    grouped_bar(
        df,
        metric="throughput_jobs_per_step",
        ylabel="Processed jobs per timestep",
        title="Throughput by static PPO scenario",
        filename="throughput_manual_grid_llm.png",
    )
    grouped_bar(
        df,
        metric="job_completion_rate",
        ylabel="Completion rate",
        title="Completion rate by static PPO scenario",
        filename="completion_manual_grid_llm.png",
    )
    grouped_bar(
        df,
        metric="training_time_seconds",
        ylabel="Training time in seconds",
        title="Training time by static PPO scenario",
        filename="training_time_manual_grid_llm.png",
    )
    grouped_bar(
        df,
        metric="reward_convergence_timestep",
        ylabel="Estimated convergence timestep",
        title="Estimated convergence timestep by static PPO scenario",
        filename="convergence_timestep_manual_grid_llm.png",
    )

    jobs_grouped(df, "Manual reward", "jobs_flow_manual.png")
    jobs_grouped(df, "Grid best", "jobs_flow_grid_best.png")
    jobs_grouped(df, "LLM best", "jobs_flow_llm_best.png")

    print("Plots saved in:", PLOT_DIR)
    print("Combined CSV:", PLOT_DIR / "combined_best_results.csv")


if __name__ == "__main__":
    main()
