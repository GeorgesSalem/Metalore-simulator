#!/bin/bash
set -Eeuo pipefail
umask 077
PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
cd "$PROJECT_DIR"

if [[ "$(hostname)" != "big20" ]]; then
  echo "ERROR: connect first with OAR_JOB_ID=1304136 oarsh big20"
  exit 1
fi
source "$PROJECT_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

DATASET="$PROJECT_DIR/cluster_runs/large/summary_live.csv"
[[ -f "$DATASET" ]] || DATASET="$PROJECT_DIR/clean_dataset_2209.csv"

python -u "$PROJECT_DIR/cluster/main_dynamic_big20/train_main_dynamic_ppo_big20.py" \
  --dataset "$DATASET" \
  --eta 0.8 --c2 -1 \
  --quick-test \
  --start-method fork \
  --device cpu \
  --output-dir main_dynamic_ppo_big20_test
