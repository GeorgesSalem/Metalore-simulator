#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-main_dynamic_ppo_big20}"

cd "$PROJECT_DIR"

if [[ "$(hostname)" != "big20" ]]; then
  echo "ERROR: This full run must be launched inside the reserved big20 node."
  echo "Enter it first with: OAR_JOB_ID=1304136 oarsh big20"
  exit 1
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: virtual environment not found: $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"

# Prevent every simulator worker from spawning its own BLAS thread pool.
# 24 environment workers + 12 PyTorch threads use the 48-core node safely.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

DATASET=""
for candidate in \
  "$PROJECT_DIR/cluster_runs/large/summary_live.csv" \
  "$PROJECT_DIR/clean_dataset_2209.csv" \
  "$PROJECT_DIR/clean_dataset_2209(1).csv"; do
  if [[ -f "$candidate" ]]; then
    DATASET="$candidate"
    break
  fi
done

if [[ -z "$DATASET" ]]; then
  echo "ERROR: no dataset was found."
  echo "Expected cluster_runs/large/summary_live.csv or clean_dataset_2209.csv"
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/cluster/main_dynamic_big20/train_main_dynamic_ppo_big20.py" ]]; then
  echo "ERROR: train_main_dynamic_ppo_big20.py is missing from $PROJECT_DIR"
  exit 1
fi

mkdir -p "$PROJECT_DIR/$OUTPUT_DIR" "$PROJECT_DIR/results_backups"
chmod -R go-rwx "$PROJECT_DIR/$OUTPUT_DIR" "$PROJECT_DIR/results_backups" 2>/dev/null || true

echo "============================================================"
echo "ONE GENERAL PPO — BIG20"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-not-set}"
echo "Dataset: $DATASET"
echo "Output: $PROJECT_DIR/$OUTPUT_DIR"
echo "Fixed eta: 0.8"
echo "Fixed C2: -1"
echo "Training: 1,000,000 total timesteps"
echo "Episode: 1,000 steps; load changes every 50 steps"
echo "Workers: 24 environments feeding ONE PPO"
echo "============================================================"

exec python -u "$PROJECT_DIR/cluster/main_dynamic_big20/train_main_dynamic_ppo_big20.py" \
  --dataset "$DATASET" \
  --eta 0.8 \
  --c2 -1 \
  --train-timesteps 1000000 \
  --episode-steps 1000 \
  --load-hold-steps 50 \
  --num-envs 24 \
  --torch-threads 12 \
  --train-max-ues 80 \
  --train-max-sensors 40 \
  --pool-max-ues 100 \
  --pool-max-sensors 50 \
  --checkpoint-every 100000 \
  --eval-episodes-per-target 3 \
  --start-method fork \
  --device cpu \
  --resume-auto \
  --output-dir "$OUTPUT_DIR"
