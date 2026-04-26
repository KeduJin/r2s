#!/usr/bin/env bash

set -euo pipefail

if [ -f .env ]; then set -a; source .env; set +a; fi
source "$CONDA_PATH"
conda activate "$ENV_NAME"
if [ -f .env ]; then set -a; source .env; set +a; fi
cd "$PROJ_DIR"

input_path="${1:-}"
port="${2:-6006}"

if [ -z "$input_path" ]; then
  latest_run="$(ls -dt output/*-qwen-chemberta-qwen3-reaction2seq 2>/dev/null | sed -n '1p')"
  if [ -z "$latest_run" ]; then
    echo "No reaction2seq run found under output/ ."
    echo "Usage: bash scripts/train/launch_reaction2seq_tensorboard.sh [logdir|run_dir] [port]"
    exit 1
  fi
  logdir="${latest_run}/tblogs"
elif [ -d "${input_path}/tblogs" ]; then
  logdir="${input_path}/tblogs"
else
  logdir="$input_path"
fi

if [ ! -d "$logdir" ]; then
  echo "TensorBoard log directory does not exist: $logdir"
  exit 1
fi

echo "Launching TensorBoard"
echo "  logdir: $logdir"
echo "  port:   $port"
echo "Open: http://127.0.0.1:${port}"

tensorboard --logdir "$logdir" --host 0.0.0.0 --port "$port"
