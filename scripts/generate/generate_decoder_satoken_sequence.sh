#!/usr/bin/env bash
# 一键：用训练好的 decoder-only 权重各跑一次生成，并合并为三个最终文件：
#   1) direct_satoken_generation.tsv      — 直接生成的 satoken
#   2) direct_sequence_generation.tsv     — 直接生成的 sequence
#   3) sequence_from_satoken.tsv          — 从 (1) 的 satoken 中提取的纯氨基酸序列
#
# 用法:
#   bash scripts/generate/generate_decoder_satoken_sequence.sh <satoken_ckpt_dir> <sequence_ckpt_dir> [out_merge_dir]
#
# 示例:
#   bash scripts/generate/generate_decoder_satoken_sequence.sh \
#     output/2026Y_01M_01D_12h-satoken/IntervalCheckpoints/step=5000 \
#     output/2026Y_01M_01D_12h-sequence/IntervalCheckpoints/step=5000 \
#     output/my_merged_three
#
# 说明:
#   - 第 3 个参数省略时，合并结果写到项目下 output/merged_decoder_three_outputs/
#   - satoken / sequence 原始 tsv 仍在各自 checkpoint 下的子目录（默认见下方环境变量）
#   - 若改了 generation 的 yaml 名，需设环境变量 SATOKEN_OUT_SUBDIR / SEQUENCE_OUT_SUBDIR 指向实际输出文件夹
#   - 仅合并、不重新生成时: python scripts/generate/merge_decoder_outputs.py --satoken_tsv ... --sequence_tsv ... --out_dir ...

set -eo pipefail
if [ -f .env ]; then set -a; source .env; set +a; fi
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-r2s}"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "${PROJ_DIR:-}" ]; then
  cd "$PROJ_DIR"
else
  cd "$_SCRIPT_DIR/../.."
fi

CKPT_SATOKEN="${1:-}"
CKPT_SEQUENCE="${2:-}"
OUT_MERGE="${3:-output/merged_decoder_three_outputs}"
GEN_YAML="configs/generation/default.yaml"
# 与 experiments/generate.py 中 test_output_name 规则一致（reaction_custom + default.yaml）
SATOKEN_OUT_SUBDIR="${SATOKEN_OUT_SUBDIR:-reaction_custom_default_output}"
SEQUENCE_OUT_SUBDIR="${SEQUENCE_OUT_SUBDIR:-reaction_custom_sequence_default_output}"

if [ -z "$CKPT_SATOKEN" ] || [ -z "$CKPT_SEQUENCE" ]; then
  echo "用法: $0 <satoken_ckpt_dir> <sequence_ckpt_dir>"
  echo "示例: $0 output/.../IntervalCheckpoints/step=5000 output/.../IntervalCheckpoints/step=5000"
  exit 1
fi

for path in "$CKPT_SATOKEN" "$CKPT_SEQUENCE"; do
  if [ ! -d "$path" ]; then
    echo "错误: 目录不存在: $path"
    exit 1
  fi
done

echo "========== [1/2] 生成 satoken (decoder-only, SaProt) =========="
bash scripts/generate/launch_reaction2seq.sh \
  "configs/train/qwen-reaction_mat_satoken_decoder_qwen3_100M_mmseqs30.yaml" \
  "configs/test/reaction_custom.yaml" \
  "$GEN_YAML" \
  "$CKPT_SATOKEN"

echo "========== [2/3] 生成 sequence (decoder-only, dplm_150m) =========="
bash scripts/generate/launch_reaction2seq.sh \
  "configs/train/qwen-reaction_mat_sequence_decoder_qwen3_100M_mmseqs30.yaml" \
  "configs/test/reaction_custom_sequence.yaml" \
  "$GEN_YAML" \
  "$CKPT_SEQUENCE"

# 多 GPU 时 R2S 写入 sequence_output_rank0.tsv；单进程为 sequence_output.tsv
_resolve_gen_tsv() {
  local d="$1"
  if [ -f "$d/sequence_output.tsv" ]; then
    echo "$d/sequence_output.tsv"
  elif [ -f "$d/sequence_output_rank0.tsv" ]; then
    echo "$d/sequence_output_rank0.tsv"
  else
    echo ""
  fi
}

SATOKEN_DIR="$CKPT_SATOKEN/$SATOKEN_OUT_SUBDIR"
SEQUENCE_DIR="$CKPT_SEQUENCE/$SEQUENCE_OUT_SUBDIR"
SATOKEN_TSV="$(_resolve_gen_tsv "$SATOKEN_DIR")"
SEQUENCE_TSV="$(_resolve_gen_tsv "$SEQUENCE_DIR")"

echo "========== [3/3] 合并为三个最终文件 -> $OUT_MERGE =========="
if [ -z "$SATOKEN_TSV" ]; then
  echo "错误: 在目录中未找到 sequence_output.tsv 或 sequence_output_rank0.tsv: $SATOKEN_DIR"
  echo "若输出目录名不同，请设置 SATOKEN_OUT_SUBDIR=实际子目录名"
  exit 1
fi
if [ -z "$SEQUENCE_TSV" ]; then
  echo "错误: 在目录中未找到 sequence_output.tsv 或 sequence_output_rank0.tsv: $SEQUENCE_DIR"
  echo "若输出目录名不同，请设置 SEQUENCE_OUT_SUBDIR=实际子目录名"
  exit 1
fi

python scripts/generate/merge_decoder_outputs.py \
  --satoken_tsv "$SATOKEN_TSV" \
  --sequence_tsv "$SEQUENCE_TSV" \
  --out_dir "$OUT_MERGE"

echo "完成。"
echo "  原始 tsv: $SATOKEN_TSV"
echo "           $SEQUENCE_TSV"
echo "  三文件合并目录: $OUT_MERGE/"
