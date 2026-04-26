# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR

_config=$1
_test_config=$2
generation_config=$3
_DIR=$4

if [ -z "$_DIR" -o -z "$_config" -o -z "$_test_config" -o -z "$generation_config" ]; then
    echo "Usage: $0 <config> <test_config> <generation_config> <directory>" 
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$_DIR" ]; then
  echo "错误：目录 '$_DIR' 不存在。"
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

if [ -z "$generation_config" ]; then
  echo "错误：generation_config '$_generation_config' 不存在。"
  exit 1
fi

for ckpt_path in $(find $_DIR/IntervalCheckpoints/ -type d -name "step=*" | awk -F'=' '$2 % 20000 == 0' | sort -t '=' -k2 -g); do
    echo "Running ckpt_path: $ckpt_path"
    if [ -f $ckpt_path/pytorch_model.bin ]; then
        echo "pytorch_model.bin exists"
    else
        echo "pytorch_model.bin not exists, creating it"
        bash scripts/others/gather_ds_model.sh $ckpt_path
    fi
    
    _test_config_name=$(basename $_test_config .yaml)
    _generation_config_name=$(basename $generation_config .yaml)
    # if exists ckpt_path/${_test_config_name}_${_generation_config_name}_output, then skip
    if [ -d $ckpt_path/${_test_config_name}_${_generation_config_name}_output ]; then
        echo "${_test_config_name}_${_generation_config_name}_output already exists, skipping"
        continue
    fi
    accelerate launch \
        --config_file accelerate_config/default.yaml \
        experiments/test.py \
        --config $_config \
        --test_config $_test_config \
        --generation_config $generation_config \
        --ckpt_path $ckpt_path
    
    # if exists ckpt_path/${_test_config_name}_output/esmfold_results, then skip
    if [ -d $ckpt_path/${_test_config_name}_${_generation_config_name}_output/esmfold_results ]; then
        echo "${_test_config_name}_${_generation_config_name}_output/esmfold_results already exists, skipping"
        continue
    fi

    # for other metrics calculation
    bash scripts/evaluation/run_all.sh $ckpt_path/${_test_config_name}_${_generation_config_name}_output
done