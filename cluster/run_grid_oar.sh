#!/bin/bash
# OAR launcher for the small/current MetaLore grid.
# Put this file in: MetaLore-simulator/cluster/run_grid_oar.sh
# Submit example:
#   oarsub -l /nodes=1/core=8,walltime=2:0:0 -S ./cluster/run_grid_oar.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
VENV_DIR="$PROJECT_DIR/.venv"
RESULTS_BACKUP_DIR="$PROJECT_DIR/results_backups"

mkdir -p "$RESULTS_BACKUP_DIR"

LOG_FILE="$PROJECT_DIR/oar_run_${OAR_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== OAR JOB START =========="
date
echo "Host: $(hostname)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-manual}"
echo "Project: $PROJECT_DIR"
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
if [ -f "$PROJECT_DIR/requirements_extra.txt" ]; then
    python -m pip install -r "$PROJECT_DIR/requirements_extra.txt"
fi
if [ -f "$PROJECT_DIR/metalore/requirements_extra.txt" ]; then
    python -m pip install -r "$PROJECT_DIR/metalore/requirements_extra.txt"
fi

export PYTHONPATH="$PROJECT_DIR/metalore:$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$PROJECT_DIR/metalore/results" "$PROJECT_DIR/metalore/models"

echo "========== RUN SMALL/CURRENT GRID =========="
python -u "$PROJECT_DIR/metalore/02_grid_search_static.py"

echo "========== BACKUP RESULTS =========="
BACKUP_FILE="$RESULTS_BACKUP_DIR/results_${OAR_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" metalore/results metalore/models 2>/dev/null || true
chmod go-rwx "$BACKUP_FILE" || true

echo "Backup created: $BACKUP_FILE"
echo "========== OAR JOB END =========="
date
