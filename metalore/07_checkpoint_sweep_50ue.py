from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt

import static_reward_common as common
from static_reward_common import (
    METALORE_DIR,
    OPERATOR_INTENTION,
    train_one_model,
    evaluate_one_model,
    score_result,
)


# ============================================================
# 50 UE checkpoint/timestep sweep using SAME method as grid search
# ============================================================

METHOD_NAME = "checkpoint_sweep"

NUM_UES = 50
NUM_SENSORS = 25

# Best grid-search reward for 50 UE / 25 sensors
ETA = 0.5
C2 = -2.0

# Training budgets to test
CHECKPOINT_TIMESTEPS = [
    25_000,
    50_000,
    75_000,
    100_000,
    125_000,
    150_000,
]

# Keep same evaluation protocol
EVAL_EPISODES = 20

BASE_DIR = (
    METALORE_DIR
    / "results"
    / "reward_pipeline"
    / "static_25s"
    / "checkpoint_sweep_50ue_25s_gridstyle_eta0p5_c2m2p0"
)

MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

BASE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def set_common_settings(timesteps: int):
    """
    Force static_reward_common.py to use the current training budget.
    This makes train_one_model() behave like your grid/manual scripts.
    """
    common.TRAIN_TIMESTEPS = timesteps

    if hasattr(common, "EVAL_EPISODES"):
        common.EVAL_EPISODES = EVAL_EPISODES

    if hasattr(common, "NUM_SENSORS"):
        common.NUM_SENSORS = NUM_SENSORS


def safe_get(row: dict, key: str, default=math.nan):
    value = row.get(key, default)
    if value is None:
        return default
    return value


def run_one_budget(timesteps: int) -> dict:
    set_common_settings(timesteps)

    experiment_name = f"checkpoint_50ue_25s_eta0p5_c2m2p0_{timesteps // 1000}k"

    run_dir = BASE_DIR / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"ppo_50ue_25s_eta0p5_c2m2p0_{timesteps // 1000}k.zip"

    print("\n================================================")
    print(f"Training 50 UE / 25 sensors for {timesteps} timesteps")
    print(f"eta = {ETA}, C2 = {C2}")
    print("================================================")

    train_info = train_one_model(
        method_name=METHOD_NAME,
        experiment_name=experiment_name,
        num_ues=NUM_UES,
        eta=ETA,
        c2=C2,
        model_path=model_path,
        run_dir=run_dir,
    )

    print(f"\nEvaluating checkpoint/budget {timesteps}...")

    eval_info = evaluate_one_model(
        method_name=METHOD_NAME,
        experiment_name=experiment_name,
        num_ues=NUM_UES,
        eta=ETA,
        c2=C2,
        model_path=model_path,
        run_dir=run_dir,
    )

    row = {}

    if train_info:
        row.update(train_info)

    if eval_info:
        row.update(eval_info)

    row["checkpoint_timestep"] = timesteps
    row["train_timesteps"] = timesteps
    row["num_ues"] = NUM_UES
    row["num_sensors"] = NUM_SENSORS
    row["eta"] = ETA
    row["c2"] = C2
    row["method"] = METHOD_NAME
    row["experiment_name"] = experiment_name
    row["model_path"] = str(model_path)

    # Compute throughput if missing
    if "throughput_jobs_per_step" not in row:
        jobs_processed = safe_get(row, "jobs_processed")
        steps = safe_get(row, "steps")

        if not math.isnan(jobs_processed) and not math.isnan(steps) and steps > 0:
            row["throughput_jobs_per_step"] = jobs_processed / steps

    # Compute score if missing
    if "score" not in row:
        row["score"] = score_result(row, OPERATOR_INTENTION)

    return row


def select_best_checkpoint(df: pd.DataFrame) -> pd.Series:
    """
    Operator intention:
    user priority = minimize AoRI while keeping completion high.

    Rule:
    1. Prefer completion >= 0.99.
    2. Among those, choose lowest AoRI.
    3. Then choose lower AoSI.
    4. If no checkpoint reaches 0.99, choose highest completion first,
       then lowest AoRI.
    """

    good = df[df["job_completion_rate"] >= 0.99].copy()

    if len(good) > 0:
        return good.sort_values(
            ["mean_aori", "mean_aosi", "checkpoint_timestep"],
            ascending=[True, True, True],
        ).iloc[0]

    return df.sort_values(
        ["job_completion_rate", "mean_aori", "mean_aosi"],
        ascending=[False, True, True],
    ).iloc[0]


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str):
    if metric not in df.columns:
        print(f"Skipping {metric}: column not found.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["checkpoint_timestep"], df[metric], marker="o")
    plt.xlabel("Training timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_all(df: pd.DataFrame):
    plot_metric(
        df,
        "mean_aori",
        "Mean AoRI",
        "50 UE / 25 sensors — AoRI vs training timesteps",
        "50ue_checkpoint_aori.png",
    )

    plot_metric(
        df,
        "mean_aosi",
        "Mean AoSI",
        "50 UE / 25 sensors — AoSI vs training timesteps",
        "50ue_checkpoint_aosi.png",
    )

    plot_metric(
        df,
        "job_completion_rate",
        "Completion rate",
        "50 UE / 25 sensors — completion rate vs training timesteps",
        "50ue_checkpoint_completion.png",
    )

    plot_metric(
        df,
        "throughput_jobs_per_step",
        "Processed jobs per timestep",
        "50 UE / 25 sensors — throughput vs training timesteps",
        "50ue_checkpoint_throughput.png",
    )

    plot_metric(
        df,
        "score",
        "Score lower is better",
        "50 UE / 25 sensors — score vs training timesteps",
        "50ue_checkpoint_score.png",
    )

    plot_metric(
        df,
        "reward_convergence_timestep",
        "Estimated convergence timestep",
        "50 UE / 25 sensors — convergence estimate",
        "50ue_checkpoint_convergence.png",
    )

    # Combined AoRI / AoSI
    if "mean_aori" in df.columns and "mean_aosi" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["checkpoint_timestep"], df["mean_aori"], marker="o", label="AoRI")
        plt.plot(df["checkpoint_timestep"], df["mean_aosi"], marker="o", label="AoSI")
        plt.xlabel("Training timesteps")
        plt.ylabel("Age value")
        plt.title("50 UE / 25 sensors — AoRI and AoSI vs training timesteps")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out_path = PLOTS_DIR / "50ue_checkpoint_aori_aosi_combined.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print("Saved:", out_path)


def main():
    print("\n=== 50 UE GRID-STYLE CHECKPOINT SWEEP ===")
    print("UE:", NUM_UES)
    print("Sensors:", NUM_SENSORS)
    print("eta:", ETA)
    print("C2:", C2)
    print("Timesteps:", CHECKPOINT_TIMESTEPS)
    print("Output:", BASE_DIR)

    rows = []

    for timesteps in CHECKPOINT_TIMESTEPS:
        row = run_one_budget(timesteps)
        rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("checkpoint_timestep")

        summary_csv = BASE_DIR / "checkpoint_summary.csv"
        df.to_csv(summary_csv, index=False)

        print("\nCurrent checkpoint summary:")
        show_cols = [
            "checkpoint_timestep",
            "mean_aori",
            "mean_aosi",
            "job_completion_rate",
            "throughput_jobs_per_step",
            "score",
            "reward_convergence_timestep",
            "training_time_seconds",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        print(df[show_cols].round(4).to_string(index=False))

    df = pd.DataFrame(rows).sort_values("checkpoint_timestep")
    summary_csv = BASE_DIR / "checkpoint_summary.csv"
    df.to_csv(summary_csv, index=False)

    plot_all(df)

    best = select_best_checkpoint(df)

    best_txt = BASE_DIR / "best_checkpoint.txt"
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("=== BEST CHECKPOINT / TRAINING BUDGET ===\n")
        f.write(f"UE = {NUM_UES}\n")
        f.write(f"Sensors = {NUM_SENSORS}\n")
        f.write(f"eta = {ETA}\n")
        f.write(f"C2 = {C2}\n")
        f.write(f"Best training timesteps = {int(best['checkpoint_timestep'])}\n")
        f.write(f"Mean AoRI = {best['mean_aori']:.4f}\n")
        f.write(f"Mean AoSI = {best['mean_aosi']:.4f}\n")
        f.write(f"Completion rate = {best['job_completion_rate']:.4f}\n")

        if "throughput_jobs_per_step" in best:
            f.write(f"Throughput jobs per step = {best['throughput_jobs_per_step']:.4f}\n")

        if "score" in best:
            f.write(f"Score = {best['score']:.4f}\n")

        if "reward_convergence_timestep" in best:
            f.write(f"Reward convergence timestep = {int(best['reward_convergence_timestep'])}\n")

        if "model_path" in best:
            f.write(f"Model path = {best['model_path']}\n")

    print("\n=== FINAL CHECKPOINT RESULTS ===")
    show_cols = [
        "checkpoint_timestep",
        "mean_aori",
        "mean_aosi",
        "job_completion_rate",
        "throughput_jobs_per_step",
        "score",
        "reward_convergence_timestep",
        "training_time_seconds",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].round(4).to_string(index=False))

    print("\n=== BEST CHECKPOINT / TRAINING BUDGET ===")
    print(f"Best timesteps: {int(best['checkpoint_timestep'])}")
    print(f"AoRI: {best['mean_aori']:.4f}")
    print(f"AoSI: {best['mean_aosi']:.4f}")
    print(f"Completion: {best['job_completion_rate']:.4f}")

    if "score" in best:
        print(f"Score: {best['score']:.4f}")

    if "reward_convergence_timestep" in best:
        print(f"Convergence timestep: {int(best['reward_convergence_timestep'])}")

    print("\nSaved summary:")
    print(summary_csv)

    print("\nSaved best checkpoint:")
    print(best_txt)

    print("\nSaved plots in:")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()