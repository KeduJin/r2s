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
# default to online logging; allow callers to override with WANDB_MODE=offline
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
  export WANDB_API_KEY=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])" 2>/dev/null)
fi

# export CUDA_VISIBLE_DEVICES=0
accelerate launch \
  --config_file accelerate_config/ds_zero2.yaml\
  experiments/train.py \
  --config "$1" "${@:2}"