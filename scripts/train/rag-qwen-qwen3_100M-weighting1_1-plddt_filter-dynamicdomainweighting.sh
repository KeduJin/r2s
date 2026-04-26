# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR
accelerate launch \
  --config_file accelerate_config/ds_zero2.yaml\
  experiments/train.py \
  --config configs/train/rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting.yaml\
  --continue_training output/2026Y_01M_25D_15h-rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting/IntervalCheckpoints/step=130000_date-01-26 2>&1 | tee \
  output/SlurmLogs/rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting_continue_training.log
