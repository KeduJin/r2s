# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR
accelerate launch \
  --config_file accelerate_config/ds_zero2_bf_autocastfalse.yaml\
  experiments/train.py \
  --config configs/train/qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter-ds_autocase_false.yaml 2>&1 | tee \
  output/SlurmLogs/qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter-ds_autocase_false.log


# accelerate launch \
#   --config_file accelerate_config/ds_zero2.yaml\
#   experiments/train.py \
#   --config configs/train/qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter.yaml \
#   --continue_training output/2025Y_12M_08D_22h-qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter/IntervalCheckpoints/step=150000_date-12-09