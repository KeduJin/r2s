# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR

_config=${1:-configs/train/qwen-chemberta-qwen3-reaction2seq.yaml}

bash scripts/train/train.sh "$_config"
