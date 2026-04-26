cd ~/fengyuan/Pretrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export WANDB_API_KET=local-30ae42b6c6f983f1f6623b6573e38f124e08160e
export NUM_NODES=1

accelerate launch \
  --config_file accelerate_config/ds_zero2.yaml\
  experiments/train.py \
  --config configs/train/dplm-esm2_35M-dplm_150M-weighting5_1.yaml 2>&1 | tee output/SlurmLogs/dplm-esm2_35M-dplm_150M-weighting5_1.log