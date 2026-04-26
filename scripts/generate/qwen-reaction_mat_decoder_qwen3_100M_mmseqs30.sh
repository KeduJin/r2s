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

task_name="${1:-satoken}"
ckpt_path="$2"
test_config_path="configs/test/reaction_custom.yaml"
generation_config_path="configs/generation/default.yaml"

case "$task_name" in
  satoken)
    config_path="configs/train/qwen-reaction_mat_satoken_decoder_qwen3_100M_mmseqs30.yaml"
    ;;
  sequence)
    config_path="configs/train/qwen-reaction_mat_sequence_decoder_qwen3_100M_mmseqs30.yaml"
    ;;
  *)
    echo "Usage: $0 [satoken|sequence] <ckpt_path>"
    exit 1
    ;;
esac

if [ -z "$ckpt_path" ]; then
  echo "Usage: $0 [satoken|sequence] <ckpt_path>"
  exit 1
fi

bash scripts/generate/launch_reaction2seq.sh \
  "$config_path" \
  "$test_config_path" \
  "$generation_config_path" \
  "$ckpt_path"
