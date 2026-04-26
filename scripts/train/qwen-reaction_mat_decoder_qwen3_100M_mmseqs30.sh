# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate r2s
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "$PROJ_DIR" ]; then
  cd "$PROJ_DIR"
else
  cd "$_SCRIPT_DIR/../.."
fi

export WANDB_MODE=${WANDB_MODE:-offline}

task_name="${1:-satoken}"
case "$task_name" in
  satoken)
    config_path="configs/train/qwen-reaction_mat_satoken_decoder_qwen3_100M_mmseqs30.yaml"
    ;;
  sequence)
    config_path="configs/train/qwen-reaction_mat_sequence_decoder_qwen3_100M_mmseqs30.yaml"
    ;;
  *)
    echo "Usage: $0 [satoken|sequence]"
    exit 1
    ;;
esac

bash scripts/train/train.sh "$config_path" --run_name "$task_name"
