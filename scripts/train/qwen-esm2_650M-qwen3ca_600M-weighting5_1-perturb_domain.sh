cd ~/fengyuan/Pretrain

# set env
if [ -f .env ]; then set -a; source .env; set +a; fi
source $CONDA_PATH
conda activate $ENV_NAME
if [ -f .env ]; then set -a; source .env; set +a; fi
cd $PROJ_DIR
accelerate launch \
  --config_file accelerate_config/ds_zero2.yaml\
  experiments/train.py \
  --continue_training output/2025Y_11M_10D_16h-qwen-esm2_650M-qwen3ca_600M-weighting5_1-perturb_domain/IntervalCheckpoints/step=50000 \
  --config configs/train/qwen-esm2_650M-qwen3ca_600M-weighting5_1-perturb_domain.yaml 2>&1 | tee output/SlurmLogs/qwen-esm2_650M-qwen3ca_600M-weighting5_1-perturb_domain.log