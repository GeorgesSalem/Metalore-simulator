#!/bin/bash
# OAR launcher for the small/current MetaLore grid.
# Submit example:
#   oarsub -l /nodes=1/core=8,walltime=2:0:0 -S ./cluster/run_grid_oar.sh
#   oarsub -l /nodes=1/core=16,walltime=12:0:0 -S ./cluster/run_grid_oar.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
VENV_DIR="$PROJECT_DIR/.venv"
RESULTS_BACKUP_DIR="$PROJECT_DIR/results_backups"

mkdir -p "$RESULTS_BACKUP_DIR"

# Log everything to a persistent file in your home project folder.
LOG_FILE="$PROJECT_DIR/oar_run_${OAR_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== OAR JOB START =========="
date
echo "Host: $(hostname)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-manual}"
echo "Project: $PROJECT_DIR"
echo "Log: $LOG_FILE"

echo "========== GO TO PROJECT =========="
cd "$PROJECT_DIR"

# Protect your project/results from other normal users.
chmod -R go-rwx "$PROJECT_DIR" || true

# Create venv only if it does not exist.
echo "========== PYTHON ENV =========="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Avoid each PPO process using too many hidden BLAS/OpenMP threads.
# For the normal sequential script, 1-4 is okay. Keep 1 for reproducibility.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make sure result directories exist in HOME, not /tmp.
mkdir -p "$PROJECT_DIR/metalore/results" "$PROJECT_DIR/metalore/models"

echo "========== RUN SMALL/CURRENT GRID =========="
# Your current script already saves partial CSV after each eta/c2/UE combination.
# For a first test, edit DRY_RUN_FIRST_N = 4 inside 02_grid_search_static.py before submitting.
python -u 02_grid_search_static.py

echo "========== BACKUP RESULTS =========="
# Create a compressed backup at the end. If the job is killed by walltime,
# partial CSVs already written inside metalore/results remain saved.
BACKUP_FILE="$RESULTS_BACKUP_DIR/results_${OAR_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" metalore/results metalore/models 2>/dev/null || true
chmod go-rwx "$BACKUP_FILE" || true

echo "Backup created: $BACKUP_FILE"
echo "========== OAR JOB END =========="
date
