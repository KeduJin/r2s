# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR
accelerate launch \
  --config_file accelerate_config/ds_zero2.yaml\
  experiments/train.py \
  --continue_training output/2025Y_10M_10D_22h-gpt-Catedesm2_35M-gpt2_114M-weighting1_1-shuffledomain/IntervalCheckpoints/step=25000\
  --config configs/train/gpt-Catedesm2_35M-gpt2_114M-weighting1_1-shuffledomain.yaml 2>&1 | tee output/SlurmLogs/gpt-Catedesm2_35M-gpt2_114M-weighting1_1-shuffledomain.log