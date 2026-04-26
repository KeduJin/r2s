# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR
accelerate launch \
  --config_file accelerate_config/ds_zero2.yaml\
  experiments/train.py \
  --config configs/train/rag-qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter-dynamicdomainweighting.yaml 2>&1 | tee \
  output/SlurmLogs/rag-qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter-dynamicdomainweighting.log


# accelerate launch \
#   --config_file accelerate_config/ds_zero2.yaml\
#   experiments/train.py \
#   --config configs/train/qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter.yaml \
#   --continue_training output/2025Y_12M_08D_22h-qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter/IntervalCheckpoints/step=150000_date-12-09