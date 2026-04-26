#!/bin/bash
hostname=`hostname`
rank_id=$(awk '/'"${hostname}"'/ {print NR-1}' ${1})
config_path=${2}
cd /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env
export NUM_NODES=2
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export WANDB_MODE=offline
export WANDB_API_KET=local-30ae42b6c6f983f1f6623b6573e38f124e08160e
export MASTER_ADDR=$(head -n 1 $1)
export MASTER_PORT=29500
echo "rank_id: $rank_id, hostname: $hostname, MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"


config_name=${config_path##*/}
config_name=${config_name%.*}

accelerate launch \
  --config_file accelerate_config/ds_zero2_for_2nodes.yaml\
  --machine_rank ${rank_id}\
  --main_process_ip ${MASTER_ADDR}\
  --main_process_port ${MASTER_PORT}\
  experiments/train.py \
  --config $config_path 2>&1 | tee \
  output/SlurmLogs/${config_name}_${rank_id}.log

