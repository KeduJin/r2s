source ~/miniconda3/etc/profile.d/conda.sh

conda activate foldflow-env
accelerate launch analysis/esmfold/run_ddp.py --test_output_path $1