#!/usr/bin/env bash
# 从 DeepSpeed ZeRO checkpoint 目录合并出 pytorch_model.bin，供非 DeepSpeed 的 generate / 单卡加载使用。
# 用法: bash scripts/others/gather_ds_model.sh <checkpoint_dir>
# 依赖: 当前可执行的 python（建议已 conda activate r2s），且能 import deepspeed
set -e
CKPT_PATH="${1:?用法: $0 <checkpoint_dir>}"
if [ ! -d "$CKPT_PATH" ]; then
  echo "错误: 目录不存在: $CKPT_PATH"
  exit 1
fi

_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_ZERO_TO_FP32="$_SCRIPT_DIR/zero_to_fp32.py"
if [ ! -f "$_ZERO_TO_FP32" ]; then
  echo "错误: 未找到 $_ZERO_TO_FP32"
  exit 1
fi

if [ -f "$CKPT_PATH/pytorch_model.bin" ]; then
  echo "pytorch_model.bin 已存在，跳过: $CKPT_PATH"
  exit 0
fi

if [ -f /opt/conda/etc/profile.d/conda.sh ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${ENV_NAME:-r2s}"
fi

cd "$CKPT_PATH"
echo "Merging DeepSpeed shards -> pytorch_model.bin in $(pwd)"
python "$_ZERO_TO_FP32" . pytorch_model.bin
echo "Done: $CKPT_PATH/pytorch_model.bin"
