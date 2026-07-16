#!/bin/bash
set -u
PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
cd "$PROJECT_DIR" || exit 1

echo "Host: $(hostname)"
echo "Date: $(date)"
echo

PID_FILE="$PROJECT_DIR/main_dynamic_ppo_big20.pid"
LOG_PATH_FILE="$PROJECT_DIR/main_dynamic_ppo_big20.logpath"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if ps -p "$PID" >/dev/null 2>&1; then
    echo "STATUS: RUNNING (PID $PID)"
    ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
  else
    echo "STATUS: PID $PID is not running"
  fi
else
  echo "STATUS: no PID file"
fi

echo
if [[ -f "main_dynamic_ppo_big20/run_status.json" ]]; then
  echo "run_status.json:"
  cat "main_dynamic_ppo_big20/run_status.json"
fi

echo
echo "Checkpoints:"
find "main_dynamic_ppo_big20/checkpoints" -maxdepth 1 -name '*_steps.zip' -printf '%f\n' 2>/dev/null | sort -V | tail -10

echo
if [[ -f "$LOG_PATH_FILE" ]]; then
  LOG_FILE="$(cat "$LOG_PATH_FILE")"
  echo "Last log lines from $LOG_FILE:"
  tail -60 "$LOG_FILE" 2>/dev/null || true
else
  echo "No log-path file found."
fi
