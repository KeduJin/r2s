#!/bin/bash

#SBATCH -p public-h800
#SBATCH -N 2
#SBATCH -J TED
#SBATCH --mem 1800G
#SBATCH -o output/SlurmLogs/%j.log
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# write the hostname of each node to a file
srun -N 2 -n 2 hostname > /storage/yuanfajieLab/yuanfajie/SlurmHost/JobID_${SLURM_JOBID}_Host
# start the training script
srun -N 2 -n 2 --ntasks-per-node=1 /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/scripts/others/training_scripts_for_2nodes.sh \
    /storage/yuanfajieLab/yuanfajie/SlurmHost/JobID_${SLURM_JOBID}_Host $1
