#!/usr/bin/env bash
# 一键：自定义反应文本 -> custom_reactions.pt -> MAT 原子表 ->（可选）双 decoder-only 生成并合并三份 TSV
#
# 「完整流程」: 建数据、跑 process_mat、再跑 satoken+sequence 两次生成 + merge
#   bash scripts/generate/one_click_custom_reaction_generate.sh <reactions.txt> <ckpt_satoken> <ckpt_sequence> [out_merge_dir]
#
# 「仅准备数据」: 只写 .pt 与 custom_mat_*（不跑生成，可稍后用同一数据手动 launch）
#   bash scripts/generate/one_click_custom_reaction_generate.sh --prepare-only <reactions.txt> [data_dir]
#
# reactions.txt: 每行一条，格式 substrate>>product 或 substrate<TAB>product ；# 开头为注释
#
# 环境变量:
#   REACT_ZYME_DIR  ReactZyme-main 根目录，默认 <enzyme-design>/ReactZyme-main
#   MAT_CHECKPOINT  MAT 权重，默认 $REACT_ZYME_DIR/pretrained/mat.pt
#   CONDA 环境: ENV_NAME 或 r2s（与项目其它脚本一致）
#
# 不用 set -u：conda 的 deactivate 会引用可能未绑定的变量，nounset 会误报
set -eo pipefail
if [ -f .env ]; then set -a; source .env; set +a; fi
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-r2s}"
if [ -f .env ]; then set -a; source .env; set +a; fi

# Pretrain-TED-DPLM 根目录
if [ -n "${PROJ_DIR:-}" ]; then
  _PROJ="$PROJ_DIR"
else
  _PROJ="$(cd "$_SCRIPT_DIR/../.." && pwd)"
fi
cd "$_PROJ"

_ENZYME_ROOT="$(cd "$_PROJ/.." && pwd)"
REACT_ZYME_DIR="${REACT_ZYME_DIR:-$_ENZYME_ROOT/ReactZyme-main}"
if [ ! -d "$REACT_ZYME_DIR" ]; then
  echo "错误: 未找到 ReactZyme-main: $REACT_ZYME_DIR"
  echo "请设置环境变量 REACT_ZYME_DIR=你的 ReactZyme-main 绝对路径"
  exit 1
fi

DATA_DIR_DEFAULT="${REACT_DATA_DIR:-$REACT_ZYME_DIR/data}"
PT_OUT="${CUSTOM_REACTIONS_PT:-$DATA_DIR_DEFAULT/custom_reactions.pt}"
SUB_EMB_OUT="${CUSTOM_MAT_SUBSTRATE:-$DATA_DIR_DEFAULT/custom_mat_substrate_embedding.pt}"
PROD_EMB_OUT="${CUSTOM_MAT_PRODUCT:-$DATA_DIR_DEFAULT/custom_mat_product_embedding.pt}"
MAT_CHECKPOINT="${MAT_CHECKPOINT:-$REACT_ZYME_DIR/pretrained/mat.pt}"

_usage() {
  echo "用法:"
  echo "  完整（生成数据 + MAT + 双模型生成 + 三文件合并）:"
  echo "    $0 <reactions.txt> <ckpt_satoken> <ckpt_sequence> [out_merge_dir]"
  echo
  echo "  仅准备 custom_reactions.pt 与 MAT（不跑生成）:"
  echo "    $0 --prepare-only <reactions.txt> [data_dir]"
  echo
  echo "  data_dir 默认: $DATA_DIR_DEFAULT  （需与 configs/test/reaction_custom*.yaml 中路径一致）"
}

_run_prepare() {
  local rxn_file="$1"
  local data_dir="${2:-$DATA_DIR_DEFAULT}"
  if [ ! -f "$rxn_file" ]; then
    echo "错误: 找不到反应文件: $rxn_file"
    exit 1
  fi
  mkdir -p "$data_dir"
  local pt_path="$data_dir/custom_reactions.pt"
  local sub_path="$data_dir/custom_mat_substrate_embedding.pt"
  local prod_path="$data_dir/custom_mat_product_embedding.pt"

  echo "========== [1/2] 写入 $pt_path =========="
  python "$_PROJ/scripts/generate/build_custom_reactions_pt.py" \
    --out "$pt_path" \
    --from-file "$rxn_file"

  if [ ! -f "$MAT_CHECKPOINT" ]; then
    echo "错误: MAT 权重不存在: $MAT_CHECKPOINT"
    echo "请设置 MAT_CHECKPOINT=你的 mat.pt 路径，或放入 ReactZyme-main/pretrained/mat.pt"
    exit 1
  fi

  echo "========== [2/2] MAT 预计算 (process_mat) =========="
  (cd "$REACT_ZYME_DIR" && python process_mat.py \
    --input_path "$pt_path" \
    --model_path "$MAT_CHECKPOINT" \
    --substrate_output_path "$sub_path" \
    --product_output_path "$prod_path")
  echo "完成数据准备："
  echo "  $pt_path"
  echo "  $sub_path"
  echo "  $prod_path"
  echo "请确认 configs/test/reaction_custom.yaml 与 reaction_custom_sequence.yaml 中"
  echo "  test_split_path / substrate_embedding_path / product_embedding_path 与上述一致。"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  _usage
  exit 0
fi

if [ "${1:-}" = "--prepare-only" ]; then
  if [ -z "${2:-}" ]; then
    _usage
    exit 1
  fi
  _run_prepare "$2" "${3:-}"
  exit 0
fi

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  _usage
  exit 1
fi

RUN_RXN_FILE="$1"
CKPT_SATOKEN="$2"
CKPT_SEQUENCE="$3"
OUT_MERGE="${4:-output/merged_custom_generate}"

_run_prepare "$RUN_RXN_FILE" "$DATA_DIR_DEFAULT"

echo "========== [3/3] 双模型生成 + 合并三文件 =========="
bash "$_PROJ/scripts/generate/generate_decoder_satoken_sequence.sh" \
  "$CKPT_SATOKEN" \
  "$CKPT_SEQUENCE" \
  "$OUT_MERGE"

echo "全部完成。合并结果目录: $OUT_MERGE/"
