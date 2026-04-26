# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-r2s}"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "$PROJ_DIR" ]; then
  cd "$PROJ_DIR"
else
  cd "$_SCRIPT_DIR/../.."
fi

export WANDB_MODE=${WANDB_MODE:-offline}

bash scripts/train/train.sh configs/train/qwen-reaction_mat_satoken_qwen3ca_100M.yaml
