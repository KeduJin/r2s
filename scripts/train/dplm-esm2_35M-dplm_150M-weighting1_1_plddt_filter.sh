# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR
accelerate launch \
  --config_file accelerate_config/ds_zero2.yaml\
  experiments/train.py \
  --continue_training output/2025Y_12M_13D_20h-dplm-esm2_35M-dplm_150M-weighting1_1_plddt_filter/IntervalCheckpoints/step=20000 \
  --config configs/train/dplm-esm2_35M-dplm_150M-weighting1_1_plddt_filter.yaml 2>&1 | tee output/SlurmLogs/dplm-esm2_35M-dplm_150M-weighting1_1_plddt_filter.log



