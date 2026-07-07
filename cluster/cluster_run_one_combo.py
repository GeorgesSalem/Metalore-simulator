"""
Run exactly one MetaLore PPO training/evaluation combination and save it immediately.

This file is designed for cluster execution and local testing. It writes both CSV
and JSON status files so completed/failed combinations are never lost.

Example from project root:
  python cluster/cluster_run_one_combo.py --num-ues 10 --num-sensors 10 --eta 0.1 --c2 -1 --timesteps 10 --eval-episodes 1 --status-dir cluster_runs/test/status
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Robust path setup.
# Expected layout:
#   MetaLore-simulator/
#     cluster/cluster_run_one_combo.py
#     metalore/static_reward_common.py
#     metalore/02_grid_search_static.py
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
CLUSTER_DIR = THIS_FILE.parent
PROJECT_ROOT = CLUSTER_DIR.parent
METALORE_SCRIPTS_DIR = PROJECT_ROOT / "metalore"

for p in (str(METALORE_SCRIPTS_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy/path/nan values into JSON-safe values."""
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


def write_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(obj), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ues", type=int, required=True)
    parser.add_argument("--num-sensors", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--c2", type=float, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--operator-intention", default="user_priority")
    parser.add_argument("--status-dir", default="cluster_runs/test/status")
    args = parser.parse_args()

    eta_tag = str(args.eta).replace("-", "m").replace(".", "p")
    c2_tag = str(args.c2).replace("-", "m").replace(".", "p")
    method_name = f"cluster_ue{args.num_ues}_s{args.num_sensors}_eta{eta_tag}_c2{c2_tag}"

    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_csv = status_dir / f"{method_name}.status.csv"
    status_json = status_dir / f"{method_name}.status.json"
    error_file = status_dir / f"{method_name}.error.txt"

    started = time.time()
    base_row = {
        "status": "started",
        "method": method_name,
        "num_ues": args.num_ues,
        "num_sensors": args.num_sensors,
        "eta": args.eta,
        "c2": args.c2,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "operator_intention": args.operator_intention,
        "project_root": str(PROJECT_ROOT),
        "metalore_scripts_dir": str(METALORE_SCRIPTS_DIR),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "started_epoch": started,
    }
    write_json(status_json, base_row)

    try:
        import static_reward_common as common

        # Patch the shared experiment settings for this one run.
        common.NUM_SENSORS = int(args.num_sensors)
        common.TRAIN_TIMESTEPS = int(args.timesteps)
        common.EVAL_EPISODES = int(args.eval_episodes)
        common.OPERATOR_INTENTION = str(args.operator_intention)

        experiment_name = common.build_experiment_name(method_name)
        row = dict(base_row)
        row["experiment"] = experiment_name

        df = common.run_training_evaluation_for_pairs(
            method_name=method_name,
            experiment_name=experiment_name,
            pairs_by_ue={int(args.num_ues): [(float(args.eta), float(args.c2))]},
            operator_intention=str(args.operator_intention),
        )

        if not df.empty:
            row.update(df.iloc[0].to_dict())

        row["status"] = "done"
        row["finished_epoch"] = time.time()
        row["wall_time_seconds_total"] = row["finished_epoch"] - started

        pd.DataFrame([row]).to_csv(status_csv, index=False)
        write_json(status_json, row)
        write_jsonl(status_dir.parent / "results_live.jsonl", row)
        return 0

    except Exception:
        tb = traceback.format_exc()
        error_file.write_text(tb, encoding="utf-8")
        row = dict(base_row)
        row.update({
            "status": "failed",
            "finished_epoch": time.time(),
            "wall_time_seconds_total": time.time() - started,
            "error_file": str(error_file),
            "error": tb.splitlines()[-1] if tb.splitlines() else "unknown error",
            "traceback": tb,
        })
        pd.DataFrame([row]).to_csv(status_csv, index=False)
        write_json(status_json, row)
        write_jsonl(status_dir.parent / "results_live.jsonl", row)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
