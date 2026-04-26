# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate foldflow-env
# export WANDB_MODE=offline
# export WANDB_API_KET=local-30ae42b6c6f983f1f6623b6573e38f124e08160e
# export NUM_NODES=1
# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR

_config=$1
_test_config=$2
_generation_config=$3
ckpt_path=$4


_test_config_name=$(basename $_test_config .yaml)
_generation_config_name=$(basename $_generation_config .yaml)

# 检查 generation_config 中是否指定了 unique domain pieces 方法
method_suffix=""
if grep -q "generate_simple_with_unique_domain_pieces" "$_generation_config" 2>/dev/null; then
    method_suffix="_unique"
fi

if [ -z "$ckpt_path" -o -z "$_config" -o -z "$_test_config" -o -z "$_generation_config" ]; then
    echo "Usage: $0 <config> <test_config> <generation_config> <directory>" 
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$ckpt_path" ]; then
  echo "错误：目录 '$ckpt_path' 不存在。"
  exit 1
fi

if [ -z "$_config" ]; then
  echo "错误：config '$_config' 不存在。"
  exit 1
fi

if [ -z "$_test_config" ]; then
  echo "错误：test_config '$_test_config' 不存在。"
  exit 1
fi

if [ -z "$_generation_config" ]; then
  echo "错误：generation_config '$_generation_config' 不存在。"
  exit 1
fi

if [ -f $ckpt_path/pytorch_model.bin ]; then
    echo "pytorch_model.bin exists"
else
    echo "pytorch_model.bin not exists, creating it"
    bash scripts/others/gather_ds_model.sh $ckpt_path
fi

accelerate launch \
    --config_file accelerate_config/default.yaml \
    experiments/generate.py \
    --config $_config \
    --test_config $_test_config \
    --generation_config $_generation_config \
    --ckpt_path $ckpt_path


# for other metrics calculation
bash scripts/evaluation/run_all.sh $ckpt_path/${_test_config_name}_${_generation_config_name}${method_suffix}_output
