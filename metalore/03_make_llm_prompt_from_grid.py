from __future__ import annotations

from pathlib import Path

import pandas as pd

from static_reward_common import (
    METALORE_DIR,
    OPERATOR_INTENTION,
    TARGET_COMPLETION,
    build_experiment_name,
)

# ============================================================
# STEP 3: build the prompt for ChatGPT / LLM
# Run this AFTER grid search finishes.
# It creates a .txt prompt that you copy into ChatGPT.
# ============================================================

GRID_METHOD_NAME = "grid"
GRID_EXPERIMENT_NAME = build_experiment_name(GRID_METHOD_NAME)

OUT_DIR = METALORE_DIR / "results" / "reward_pipeline" / "static_25s" / GRID_EXPERIMENT_NAME
GRID_ALL_RESULTS = OUT_DIR / "grid_all_results.csv"
PROMPT_PATH = OUT_DIR / f"llm_prompt_{OPERATOR_INTENTION}.txt"

TOP_K_PER_UE = 8


def intention_description() -> str:
    if OPERATOR_INTENTION == "user_priority":
        return "User priority: minimize AoRI while keeping completion rate high."
    if OPERATOR_INTENTION == "sensor_priority":
        return "Sensor priority: minimize AoSI while keeping completion rate high."
    if OPERATOR_INTENTION == "balanced":
        return "Balanced priority: minimize AoRI and AoSI while keeping completion rate high."
    if OPERATOR_INTENTION == "fairness":
        return "Fairness priority: maximize sum log(r_i), where r_i is the UE service rate, while keeping completion rate high."
    return OPERATOR_INTENTION


def main():
    if not GRID_ALL_RESULTS.exists():
        raise FileNotFoundError(f"Grid results not found: {GRID_ALL_RESULTS}")

    df = pd.read_csv(GRID_ALL_RESULTS)
    df = df.sort_values(["num_ues", "score"])

    keep_cols = [
        "num_ues",
        "num_sensors",
        "eta",
        "c2",
        "score",
        "mean_aori",
        "mean_aosi",
        "job_completion_rate",
        "throughput_jobs_per_step",
        "jobs_generated",
        "jobs_transmitted",
        "jobs_processed",
        "fairness_log_sum",
        "training_time_seconds",
        "reward_convergence_timestep",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    parts = []
    for num_ues, group in df.groupby("num_ues"):
        parts.append(f"\nTop {TOP_K_PER_UE} grid-search results for {num_ues} UEs / 25 sensors:")
        parts.append(group.head(TOP_K_PER_UE)[keep_cols].round(4).to_csv(index=False))

    results_text = "\n".join(parts)

    prompt = f"""
You are an outer-loop reward-parameter search agent for a PPO resource-allocation model.

Context:
- PPO hyperparameters are fixed.
- Sensors are fixed to 25.
- The UE scenarios are 10, 20, 30, 40, and 50 UEs.
- Each PPO model is trained and evaluated on the same UE/sensor scenario.
- Only two reward hyperparameters can be changed:
  1. eta: synchronization discount factor, allowed range [0.1, 1.0]
  2. c2: delay penalty, allowed range [-5.0, 0.0]
- eta is config["reward"]["discount_factor"].
- c2 is config["reward"]["delay_penalty"].
- Completion target is at least {TARGET_COMPLETION}.

Operator intention:
{intention_description()}

Task:
Analyze the grid-search results below and propose LLM-based candidate reward parameters.
For each UE scenario, return 2 or 3 candidate pairs (eta, c2) that should be trained next.
Do not change PPO hyperparameters.
Do not change number of UEs or sensors.
Do not suggest values outside the allowed ranges.
Prefer candidates that improve the operator intention while keeping completion high.

Return ONLY a Python dictionary exactly in this format:

LLM_CANDIDATES = {{
    10: [(eta, c2), (eta, c2)],
    20: [(eta, c2), (eta, c2)],
    30: [(eta, c2), (eta, c2)],
    40: [(eta, c2), (eta, c2)],
    50: [(eta, c2), (eta, c2)],
}}

Grid-search results:
{results_text}
""".strip()

    PROMPT_PATH.write_text(prompt, encoding="utf-8")
    print("LLM prompt saved to:", PROMPT_PATH)
    print("Open this .txt file, copy everything, and paste it into ChatGPT.")


if __name__ == "__main__":
    main()
