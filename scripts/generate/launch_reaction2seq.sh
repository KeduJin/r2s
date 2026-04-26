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

if [ "$#" -eq 2 ]; then
    _config=$1
    _test_config=configs/test/test_reaction2seq.yaml
    _generation_config=configs/generation/default.yaml
    ckpt_path=$2
elif [ "$#" -eq 4 ]; then
    _config=$1
    _test_config=$2
    _generation_config=$3
    ckpt_path=$4
else
    echo "Usage: $0 <config> <ckpt_dir>"
    echo "   or: $0 <config> <test_config> <generation_config> <ckpt_dir>"
    exit 1
fi

if [ ! -d "$ckpt_path" ]; then
    echo "错误：目录 '$ckpt_path' 不存在。"
    exit 1
fi

if [ ! -f "$ckpt_path/pytorch_model.bin" ]; then
    echo "pytorch_model.bin not exists, creating it"
    bash scripts/others/gather_ds_model.sh "$ckpt_path"
fi

accelerate launch \
    --config_file accelerate_config/default.yaml \
    experiments/generate.py \
    --config "$_config" \
    --test_config "$_test_config" \
    --generation_config "$_generation_config" \
    --ckpt_path "$ckpt_path"
