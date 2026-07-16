#!/bin/bash
set -u
PROJECT_DIR="${PROJECT_DIR:-$HOME/MetaLore-simulator}"
PID_FILE="$PROJECT_DIR/main_dynamic_ppo_big20.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found."
  exit 1
fi
PID="$(cat "$PID_FILE")"
if ps -p "$PID" >/dev/null 2>&1; then
  echo "Sending SIGTERM to PID $PID. The Python script will save an emergency model."
  kill -TERM "$PID"
else
  echo "PID $PID is not running."
fi
