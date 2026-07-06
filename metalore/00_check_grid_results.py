import pandas as pd

from static_reward_common import (
    METALORE_DIR,
    SENSOR_VALUES,
    UE_VALUES,
    build_experiment_name,
    sensor_tag,
)

GRID_METHOD_NAME = "grid_ue20_40_60_s10_20_30"
GRID_EXPERIMENT_NAME = build_experiment_name(GRID_METHOD_NAME, sensor_values=SENSOR_VALUES)
STATIC_FOLDER = f"static_{sensor_tag(sensor_values=SENSOR_VALUES)}s"

BASE_DIR = (
    METALORE_DIR
    / "results"
    / "reward_pipeline"
    / STATIC_FOLDER
    / GRID_EXPERIMENT_NAME
)

ALL_RESULTS = BASE_DIR / f"{GRID_METHOD_NAME}_all_results.csv"
BEST_RESULTS = BASE_DIR / f"{GRID_METHOD_NAME}_best_by_ue_sensor.csv"

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
print("Expected sensor values:", SENSOR_VALUES)
print("Found sensor values:", sorted(best_df["num_sensors"].unique().tolist()))

print("\n=== BEST ETA/C2 BY UE/SENSORS ===")

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
expected_best_rows = len(UE_VALUES) * len(SENSOR_VALUES)

if len(best_df) != expected_best_rows:
    errors.append(f"Expected {expected_best_rows} best rows, found {len(best_df)}")

if sorted(best_df["num_ues"].unique().tolist()) != sorted(UE_VALUES):
    errors.append("Best result file does not contain all UE values.")

if sorted(best_df["num_sensors"].unique().tolist()) != sorted(SENSOR_VALUES):
    errors.append("Best result file does not contain all sensor values.")

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
    sensors = int(row["num_sensors"])
    eta = row["eta"]
    c2 = row["c2"]
    completion = row["job_completion_rate"]
    aori = row["mean_aori"]
    aosi = row["mean_aosi"]
    conv = row.get("reward_convergence_timestep", None)

    print(f"\n{ue} UE / {sensors} sensors:")
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

print(f"\nNext step: inspect {BEST_RESULTS.name}, then compare with manual or LLM results.")
