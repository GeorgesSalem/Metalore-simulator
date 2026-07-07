"""
Parallel launcher for MetaLore grid search.

It creates combinations, runs independent Python processes, and saves:
  cluster_runs/<mode>/planned_combinations.csv
  cluster_runs/<mode>/results_live.jsonl      # appended after each combo
  cluster_runs/<mode>/results_final.json      # final JSON list
  cluster_runs/<mode>/summary_live.csv
  cluster_runs/<mode>/summary_final.csv
  cluster_runs/<mode>/logs/*.log
  cluster_runs/<mode>/status/*.status.json

Example from project root:
  python cluster/cluster_parallel_grid.py --mode test --workers 1 --timesteps 10 --eval-episodes 1
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

THIS_FILE = Path(__file__).resolve()
CLUSTER_DIR = THIS_FILE.parent
PROJECT_ROOT = CLUSTER_DIR.parent
METALORE_SCRIPTS_DIR = PROJECT_ROOT / "metalore"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True), encoding="utf-8")


def values_for_mode(mode: str):
    if mode == "test":
        return {
            "ue_values": [10],
            "sensor_values": [10],
            "eta_values": [0.1, 0.2],
            "c2_values": [-1.0, -2.0],
        }
    if mode == "small":
        return {
            "ue_values": [10, 30, 50, 70, 90],
            "sensor_values": [25],
            "eta_values": [0.5, 0.7, 0.8, 0.9],
            "c2_values": [-5.0, -4.0, -3.0, -2.0],
        }
    if mode == "large":
        return {
            "ue_values": list(range(10, 101, 10)),
            "sensor_values": list(range(10, 51, 10)),
            "eta_values": [round(x / 10, 1) for x in range(1, 10)],  # 0.1 to 0.9
            "c2_values": [-1.0, -2.0, -3.0, -4.0, -5.0],
        }
    raise ValueError(f"Unknown mode: {mode}")


def combo_name(num_ues: int, num_sensors: int, eta: float, c2: float) -> str:
    return f"ue{num_ues}_s{num_sensors}_eta{eta}_c2{c2}".replace(".", "p").replace("-", "m")


def run_combo(combo, args, logs_dir: Path, status_dir: Path) -> int:
    num_ues, num_sensors, eta, c2 = combo
    name = combo_name(num_ues, num_sensors, eta, c2)
    log_path = logs_dir / f"{name}.log"

    cmd = [
        sys.executable,
        "-u",
        str(CLUSTER_DIR / "cluster_run_one_combo.py"),
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
    pythonpath_parts = [str(METALORE_SCRIPTS_DIR), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command: " + " ".join(cmd) + "\n")
        f.write("PYTHONPATH: " + env.get("PYTHONPATH", "") + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
    return proc.returncode


def load_status_rows(status_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(status_dir.glob("*.status.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return rows


def combine_status(status_dir: Path, out_csv: Path, out_json: Path) -> None:
    rows = load_status_rows(status_dir)
    if not rows:
        # Fallback for CSV-only status files.
        files = sorted(status_dir.glob("*.status.csv"))
        frames = []
        for p in files:
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(out_csv, index=False)
            write_json(out_json, df.to_dict(orient="records"))
        return

    write_json(out_json, rows)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "small", "large"], default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--operator-intention", default="user_priority")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "cluster_runs"))
    args = parser.parse_args()

    vals = values_for_mode(args.mode)
    combos = list(itertools.product(vals["ue_values"], vals["sensor_values"], vals["eta_values"], vals["c2_values"]))

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

    plan = {
        "mode": args.mode,
        "workers": args.workers,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "operator_intention": args.operator_intention,
        "project_root": str(PROJECT_ROOT),
        "metalore_scripts_dir": str(METALORE_SCRIPTS_DIR),
        "output_dir": str(output_dir.resolve()),
        "combinations_count": len(combos),
        "values": vals,
    }
    write_json(output_dir / "run_plan.json", plan)

    print(f"Mode: {args.mode}")
    print(f"Combinations: {len(combos)}")
    print(f"Workers: {args.workers}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Planned combinations: {combo_csv}")
    print(f"JSON live file: {output_dir / 'results_live.jsonl'}")
    print(f"JSON final file: {output_dir / 'results_final.json'}")

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_combo = {executor.submit(run_combo, combo, args, logs_dir, status_dir): combo for combo in combos}
        for future in as_completed(future_to_combo):
            combo = future_to_combo[future]
            try:
                rc = future.result()
            except Exception as e:
                rc = 1
                print("FAILED", combo, "launcher exception", repr(e))
            if rc == 0:
                print("DONE", combo)
            else:
                failures += 1
                print("FAILED", combo, "returncode", rc)
            combine_status(status_dir, output_dir / "summary_live.csv", output_dir / "results_final.json")

    combine_status(status_dir, output_dir / "summary_final.csv", output_dir / "results_final.json")
    print("Final CSV summary:", output_dir / "summary_final.csv")
    print("Final JSON results:", output_dir / "results_final.json")
    print("Live JSONL results:", output_dir / "results_live.jsonl")
    print("Failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
