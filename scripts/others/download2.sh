source ~/miniconda3/etc/profile.d/conda.sh
conda activate aigc
# export WANDB_MODE=offline
# export WANDB_API_KET=local-30ae42b6c6f983f1f6623b6573e38f124e08160e
# export NUM_NODES=1

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 设置5分钟超时

COMMAND="huggingface-cli download facebook/esm2_t30_150M_UR50D "
# COMMAND="huggingface-cli download --token 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 --resume-download Qwen/Qwen3-VL-4B-Instruct"

until $COMMAND; do
        # $? 存储了上一条命令的退出码
        echo "----------------------------------------------------"
        echo "命令执行失败，退出码为 $?. 正在准备重试..."
        echo "5秒后将重新执行命令。"
        echo "----------------------------------------------------"
        sleep 5
done


