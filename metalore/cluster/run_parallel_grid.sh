#!/bin/bash
# OAR launcher for parallel MetaLore grid runs.
# Test submit:
#   oarsub -l /nodes=1/core=8,walltime=2:0:0 -S ./cluster/run_parallel_grid.sh
# Large submit after test:
#   MODE=large WORKERS=48 oarsub -l /nodes=1/core=48,walltime=24:0:0 -S ./cluster/run_parallel_grid.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
VENV_DIR="$PROJECT_DIR/.venv"
MODE="${MODE:-test}"          # test, small, or large
WORKERS="${WORKERS:-4}"       # for test use 4; for full node big16-big20 use up to 48
TIMESTEPS="${TIMESTEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
OPERATOR_INTENTION="${OPERATOR_INTENTION:-user_priority}"
RESULTS_BACKUP_DIR="$PROJECT_DIR/results_backups"

mkdir -p "$RESULTS_BACKUP_DIR"

LOG_FILE="$PROJECT_DIR/oar_parallel_${MODE}_${OAR_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== OAR PARALLEL JOB START =========="
date
echo "Host: $(hostname)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-manual}"
echo "Project: $PROJECT_DIR"
echo "Mode: $MODE"
echo "Workers: $WORKERS"
echo "Timesteps: $TIMESTEPS"
echo "Eval episodes: $EVAL_EPISODES"
echo "Log: $LOG_FILE"

cd "$PROJECT_DIR"
chmod -R go-rwx "$PROJECT_DIR" || true

echo "========== PYTHON ENV =========="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$PROJECT_DIR/metalore/results" "$PROJECT_DIR/metalore/models" "$PROJECT_DIR/cluster_runs"

echo "========== RUN PARALLEL GRID =========="
python -u cluster_parallel_grid.py \
  --mode "$MODE" \
  --workers "$WORKERS" \
  --timesteps "$TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --operator-intention "$OPERATOR_INTENTION" \
  --output-dir "$PROJECT_DIR/cluster_runs"

echo "========== BACKUP RESULTS =========="
BACKUP_FILE="$RESULTS_BACKUP_DIR/${MODE}_results_${OAR_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" cluster_runs metalore/results metalore/models 2>/dev/null || true
chmod go-rwx "$BACKUP_FILE" || true

echo "Backup created: $BACKUP_FILE"
echo "========== OAR PARALLEL JOB END =========="
date
