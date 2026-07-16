#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
cd "$PROJECT_DIR"

if [[ "$(hostname)" != "big20" ]]; then
  echo "ERROR: connect to big20 first: OAR_JOB_ID=1304136 oarsh big20"
  exit 1
fi

PID_FILE="$PROJECT_DIR/main_dynamic_ppo_big20.pid"
LOG_PATH_FILE="$PROJECT_DIR/main_dynamic_ppo_big20.logpath"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && ps -p "$OLD_PID" >/dev/null 2>&1; then
    echo "Training is already running with PID $OLD_PID"
    exit 1
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROJECT_DIR/main_dynamic_ppo_big20/logs"
LOG_FILE="$PROJECT_DIR/main_dynamic_ppo_big20/logs/run_${STAMP}.log"

echo "$LOG_FILE" > "$LOG_PATH_FILE"
nohup bash "$PROJECT_DIR/cluster/main_dynamic_big20/run_main_dynamic_big20.sh" \
  > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 2
if ps -p "$PID" >/dev/null 2>&1; then
  echo "Started successfully."
  echo "PID: $PID"
  echo "Log: $LOG_FILE"
  echo "You may close the terminal."
  echo "Check later with: ./cluster/main_dynamic_big20/check_main_dynamic_big20.sh"
else
  echo "The process stopped immediately. Check: $LOG_FILE"
  tail -80 "$LOG_FILE" || true
  exit 1
fi
