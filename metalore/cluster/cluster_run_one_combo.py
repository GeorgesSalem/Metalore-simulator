"""
Run exactly one MetaLore PPO training/evaluation combination and save it immediately.
This file is useful for large-grid cluster runs, because each process writes its own
result folder instead of many workers fighting over the same CSV.

Example:
  python cluster_run_one_combo.py --num-ues 10 --num-sensors 10 --eta 0.7 --c2 -2 --timesteps 100000 --eval-episodes 20
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

import static_reward_common as common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ues", type=int, required=True)
    parser.add_argument("--num-sensors", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--c2", type=float, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--operator-intention", default=common.OPERATOR_INTENTION)
    parser.add_argument("--status-dir", default="cluster_status")
    args = parser.parse_args()

    # Patch the shared experiment settings for this one run.
    common.NUM_SENSORS = int(args.num_sensors)
    common.TRAIN_TIMESTEPS = int(args.timesteps)
    common.EVAL_EPISODES = int(args.eval_episodes)
    common.OPERATOR_INTENTION = str(args.operator_intention)

    eta_tag = common.tag_float(args.eta)
    c2_tag = common.tag_float(args.c2)

    method_name = (
        f"cluster_ue{args.num_ues}_s{args.num_sensors}_"
        f"eta{eta_tag}_c2{c2_tag}"
    )
    experiment_name = common.build_experiment_name(method_name)

    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / f"{method_name}.status.csv"
    error_file = status_dir / f"{method_name}.error.txt"

    try:
        df = common.run_training_evaluation_for_pairs(
            method_name=method_name,
            experiment_name=experiment_name,
            pairs_by_ue={int(args.num_ues): [(float(args.eta), float(args.c2))]},
            operator_intention=str(args.operator_intention),
        )
        row = {
            "status": "done",
            "num_ues": args.num_ues,
            "num_sensors": args.num_sensors,
            "eta": args.eta,
            "c2": args.c2,
            "timesteps": args.timesteps,
            "eval_episodes": args.eval_episodes,
            "experiment": experiment_name,
        }
        if not df.empty:
            row.update(df.iloc[0].to_dict())
        pd.DataFrame([row]).to_csv(status_file, index=False)
        return 0
    except Exception:
        error_file.write_text(traceback.format_exc(), encoding="utf-8")
        pd.DataFrame([{
            "status": "failed",
            "num_ues": args.num_ues,
            "num_sensors": args.num_sensors,
            "eta": args.eta,
            "c2": args.c2,
            "timesteps": args.timesteps,
            "eval_episodes": args.eval_episodes,
            "experiment": experiment_name,
            "error_file": str(error_file),
        }]).to_csv(status_file, index=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
