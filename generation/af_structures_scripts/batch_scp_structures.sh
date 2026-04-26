#!/bin/bash

# 批量从远程服务器下载 AlphaFold 结构文件并自动解压

# 配置参数
UIDS_FILE="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/generation/af_structures_scripts/uids.txt"
REMOTE_HOST="TempCluster_yungu"
REMOTE_DIR="/ssd/share-data/gzfile"
LOCAL_DIR="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/generation/af_structures"

# 创建本地目录（如果不存在）
mkdir -p "$LOCAL_DIR"

# 统计总数
total=$(grep -c . "$UIDS_FILE")
current=0
success_count=0
fail_count=0

echo "开始下载并解压 $total 个结构文件..."

# 读取 uids.txt 文件并逐行处理
while IFS= read -r uid || [ -n "$uid" ]; do
    # 跳过空行
    if [ -z "$uid" ]; then
        continue
    fi

    current=$((current + 1))

    # 构造文件名
    filename="AF-${uid}-F1-model_v4.cif.gz"
    remote_path="${REMOTE_HOST}:${REMOTE_DIR}/${filename}"

    # 显示进度
    printf "\r进度: %d/%d (成功: %d, 失败: %d)" "$current" "$total" "$success_count" "$fail_count"

    # 执行 scp 命令（静默模式）
    if scp -q "$remote_path" "$LOCAL_DIR" 2>/dev/null; then
        # 解压文件（静默模式）
        gz_file="${LOCAL_DIR}/${filename}"
        if gunzip -q "$gz_file" 2>/dev/null; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
            echo ""
            echo "✗ 解压失败: $filename"
        fi
    else
        fail_count=$((fail_count + 1))
        echo ""
        echo "✗ 下载失败: $filename"
    fi

done < "$UIDS_FILE"

# 输出最终统计
echo ""
echo "=========================================="
echo "完成！总数: $total | 成功: $success_count | 失败: $fail_count"
echo "=========================================="
