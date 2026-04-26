source ~/miniconda3/etc/profile.d/conda.sh
conda activate foldflow-env
# export WANDB_MODE=offline
# export WANDB_API_KET=local-30ae42b6c6f983f1f6623b6573e38f124e08160e
# export NUM_NODES=1

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 设置5分钟超时

# COMMANDS=(
#   "huggingface-cli download facebook/esm2_t33_650M_UR50D"
#   "huggingface-cli download facebook/esm2_t12_35M_UR50D"
#   "huggingface-cli download airkingbd/dplm_150m"
#   "huggingface-cli download airkingbd/dplm_650m"
#   "huggingface-cli download --token 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 Qwen/Qwen3-0.6B --force-download"
# )
COMMANDS=(
  "huggingface-cli download nwliu/ProDVa-CAMEO"
  "huggingface-cli download nwliu/ProDVa-Molinst-SwissProtCLAP"
)


for cmd in "${COMMANDS[@]}"; do
  echo "Executing: $cmd"
  until eval "$cmd"; do
    echo "----------------------------------------------------"
    echo "命令执行失败，退出码为 $?. 正在准备重试..."
    echo "5秒后将重新执行命令。"
    echo "----------------------------------------------------"
    sleep 5
  done
done


