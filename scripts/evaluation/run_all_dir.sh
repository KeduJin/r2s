# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR


_DIR=$1

if [ -z "$_DIR" ]; then
    echo "Usage: $0 <directory>" 
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$_DIR" ]; then
  echo "错误：目录 '$_DIR' 不存在。"
  exit 1
fi


for ckpt_path in $_DIR/IntervalCheckpoints/step=*0000*; do
    echo "Runningckpt_path: $ckpt_path/test_shuffledomain_output"
    bash scripts/evaluation/run_all.sh $ckpt_path/test_shuffledomain_output
done