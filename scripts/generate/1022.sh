source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env

accelerate launch \
    --config_file accelerate_config/default.yaml \
    experiments/generate.py \
    --config configs/train/qwen-esm2_35M-qwen3ca_100M-weighting5_1.yaml \
    --test_config configs/test/generatoin-1022.yaml \
    --ckpt_path output/2025Y_10M_16D_23h-qwen-esm2_35M-qwen3ca_100M-weighting5_1/IntervalCheckpoints/step=200000