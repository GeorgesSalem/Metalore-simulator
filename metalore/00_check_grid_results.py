from pathlib import Path
import pandas as pd

from static_reward_common import (
    METALORE_DIR,
    UE_VALUES,
    NUM_SENSORS,
    OPERATOR_INTENTION,
    TRAIN_TIMESTEPS,
)

BASE_DIR = (
    METALORE_DIR
    / "results"
    / "reward_pipeline"
    / "static_25s"
    / f"grid_static_25s_{OPERATOR_INTENTION}_{TRAIN_TIMESTEPS // 1000}k"
)

ALL_RESULTS = BASE_DIR / "grid_all_results.csv"
BEST_RESULTS = BASE_DIR / "grid_best_by_ue.csv"

print("\n=== CHECKING GRID SEARCH RESULTS ===")
print("Folder:", BASE_DIR)

if not BASE_DIR.exists():
    raise FileNotFoundError(f"Folder not found: {BASE_DIR}")

if not ALL_RESULTS.exists():
    raise FileNotFoundError(f"Missing: {ALL_RESULTS}")

if not BEST_RESULTS.exists():
    raise FileNotFoundError(f"Missing: {BEST_RESULTS}")

all_df = pd.read_csv(ALL_RESULTS)
best_df = pd.read_csv(BEST_RESULTS)

print("\nTotal grid rows:", len(all_df))
print("Best rows:", len(best_df))
print("Expected UE values:", UE_VALUES)
print("Found UE values:", sorted(best_df["num_ues"].unique().tolist()))

print("\n=== BEST ETA/C2 BY UE ===")

cols = [
    "num_ues",
    "num_sensors",
    "eta",
    "c2",
    "mean_aori",
    "mean_aosi",
    "job_completion_rate",
    "throughput_jobs_per_step",
    "jobs_generated",
    "jobs_transmitted",
    "jobs_processed",
    "training_time_seconds",
    "steps_per_second",
    "reward_convergence_timestep",
    "score",
]

cols = [c for c in cols if c in best_df.columns]
print(best_df[cols].round(4).to_string(index=False))

print("\n=== QUICK CHECK ===")

errors = []

if len(best_df) != len(UE_VALUES):
    errors.append(f"Expected {len(UE_VALUES)} best rows, found {len(best_df)}")

if sorted(best_df["num_ues"].unique().tolist()) != sorted(UE_VALUES):
    errors.append("Best result file does not contain all UE values.")

if not (best_df["num_sensors"] == NUM_SENSORS).all():
    errors.append("Some rows do not have 25 sensors.")

missing_cols = [c for c in cols if c not in best_df.columns]
if missing_cols:
    errors.append(f"Missing columns: {missing_cols}")

if errors:
    print("\nProblems found:")
    for e in errors:
        print("-", e)
else:
    print("\nGrid result structure is OK.")

print("\n=== INTERPRETATION ===")

for _, row in best_df.iterrows():
    ue = int(row["num_ues"])
    eta = row["eta"]
    c2 = row["c2"]
    completion = row["job_completion_rate"]
    aori = row["mean_aori"]
    aosi = row["mean_aosi"]
    conv = row.get("reward_convergence_timestep", None)

    print(f"\n{ue} UE / {NUM_SENSORS} sensors:")
    print(f"- Best eta = {eta}")
    print(f"- Best C2 = {c2}")
    print(f"- Completion = {completion * 100:.2f}%")
    print(f"- AoRI = {aori:.4f}")
    print(f"- AoSI = {aosi:.4f}")

    if conv is not None:
        print(f"- Convergence timestep = {conv}")

    if completion >= 0.99 and aori <= 1.2 and aosi <= 1.2:
        print("- Result looks very good.")
    elif completion >= 0.98:
        print("- Result is acceptable, but maybe can improve.")
    else:
        print("- Result is still weak. This UE scenario may need longer training or better reward.")

print("\nNext step: compare grid_best_by_ue.csv with the manual result.")