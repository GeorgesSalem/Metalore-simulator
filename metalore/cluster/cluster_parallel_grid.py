"""
Parallel launcher for the large MetaLore grid.
It creates all combinations, runs many independent Python processes, and each
combination saves its result immediately.

Small test:
  python cluster_parallel_grid.py --mode test --workers 4

Large grid:
  python cluster_parallel_grid.py --mode large --workers 48
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def values_for_mode(mode: str):
    if mode == "test":
        # Very small: use this first to check venv, code, saving, and logs.
        return {
            "ue_values": [10],
            "sensor_values": [10],
            "eta_values": [0.1, 0.2],
            "c2_values": [-1.0, -2.0],
        }
    if mode == "small":
        # Similar to your current small experiment but with fixed sensors=25.
        return {
            "ue_values": [10, 30, 50, 70, 90],
            "sensor_values": [25],
            "eta_values": [0.5, 0.7, 0.8, 0.9],
            "c2_values": [-5.0, -4.0, -3.0, -2.0],
        }
    if mode == "large":
        # Your planned large scenario.
        return {
            "ue_values": list(range(10, 101, 10)),
            "sensor_values": list(range(10, 51, 10)),
            "eta_values": [round(x / 10, 1) for x in range(1, 10)],
            "c2_values": [-1.0, -2.0, -3.0, -4.0, -5.0],
        }
    raise ValueError(f"Unknown mode: {mode}")


def run_combo(combo, args, logs_dir: Path, status_dir: Path) -> int:
    num_ues, num_sensors, eta, c2 = combo
    name = f"ue{num_ues}_s{num_sensors}_eta{eta}_c2{c2}".replace(".", "p").replace("-", "m")
    log_path = logs_dir / f"{name}.log"

    cmd = [
        sys.executable,
        "-u",
        "cluster/cluster_run_one_combo.py",
        "--num-ues", str(num_ues),
        "--num-sensors", str(num_sensors),
        "--eta", str(eta),
        "--c2", str(c2),
        "--timesteps", str(args.timesteps),
        "--eval-episodes", str(args.eval_episodes),
        "--operator-intention", args.operator_intention,
        "--status-dir", str(status_dir),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def combine_status(status_dir: Path, out_csv: Path) -> None:
    files = sorted(status_dir.glob("*.status.csv"))
    if not files:
        return
    frames = []
    for p in files:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "small", "large"], default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--operator-intention", default="user_priority")
    parser.add_argument("--output-dir", default="cluster_runs")
    args = parser.parse_args()

    vals = values_for_mode(args.mode)
    combos = list(itertools.product(
        vals["ue_values"], vals["sensor_values"], vals["eta_values"], vals["c2_values"]
    ))

    output_dir = Path(args.output_dir) / args.mode
    logs_dir = output_dir / "logs"
    status_dir = output_dir / "status"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    combo_csv = output_dir / "planned_combinations.csv"
    with combo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_ues", "num_sensors", "eta", "c2"])
        writer.writerows(combos)

    print(f"Mode: {args.mode}")
    print(f"Combinations: {len(combos)}")
    print(f"Workers: {args.workers}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Planned combinations: {combo_csv}")

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_combo = {
            executor.submit(run_combo, combo, args, logs_dir, status_dir): combo
            for combo in combos
        }
        for future in as_completed(future_to_combo):
            combo = future_to_combo[future]
            rc = future.result()
            if rc == 0:
                print("DONE", combo)
            else:
                failures += 1
                print("FAILED", combo, "returncode", rc)
            combine_status(status_dir, output_dir / "summary_live.csv")

    combine_status(status_dir, output_dir / "summary_final.csv")
    print("Final summary:", output_dir / "summary_final.csv")
    print("Failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
