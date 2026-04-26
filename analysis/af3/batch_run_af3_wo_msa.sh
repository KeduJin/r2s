#!/bin/bash

# 批量运行 AF3 结构预测脚本
# 用法: bash analysis/af3/batch_run_af3.sh
cd ~/fengyuan/Pretrain
unset LD_LIBRARY_PATH
unset PYTHONPATH
# 定义四个模型及其对应的路径
declare -A models=(
    ["encoder-decoder"]="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/output/2025Y_12M_28D_17h-qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter"
    ["decoder-only"]="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/output/2025Y_12M_08D_01h-qwen-qwen3_100M-weighting1_1_plddt_filter"
    ["encoder-decoder+dynamic_vocab"]="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/output/2026Y_01M_23D_16h-rag-qwen-esm2_35M-qwen3ca_100M-weighting1_1-plddt_filter-dynamicdomainweighting"
    ["decoder-only+dynamic_vocab"]="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/output/2026Y_01M_25D_15h-rag-qwen-qwen3_100M-weighting1_1-plddt_filter-dynamicdomainweighting"
)

# 定义每个模型对应的测试目录名称
declare -A test_dir_names=(
    ["encoder-decoder"]="test_plddt_filter_default_output"
    ["decoder-only"]="test_plddt_filter_default_output"
    ["encoder-decoder+dynamic_vocab"]="test_plddt_filter-bs1_default_output"
    ["decoder-only+dynamic_vocab"]="test_plddt_filter-bs1_default_output"
)

# 定义要测试的步数
steps=(10000 20000 30000 40000 50000)

# 创建日志目录
log_dir="/storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/af3/logs"
mkdir -p "$log_dir"

# 记录总开始时间
total_start_time=$(date +%s)

echo "=========================================="
echo "开始批量运行 AF3 结构预测"
echo "时间: $(date)"
echo "=========================================="

# 遍历每个模型
for model_name in "${!models[@]}"; do
    model_path="${models[$model_name]}"
    echo ""
    echo "==================== 处理模型: $model_name ===================="
    echo "模型路径: $model_path"

    # 遍历每个步数
    for step in "${steps[@]}"; do
        echo ""
        echo "---------- Step: $step ----------"

        # 根据模型名称获取对应的测试目录名称
        test_dir_name="${test_dir_names[$model_name]}"
        test_output_path="$model_path/IntervalCheckpoints/step=$step/$test_dir_name"

        if [ ! -d "$test_output_path" ]; then
            echo "警告: 未找到测试输出目录 $test_output_path,跳过"
            continue
        fi

        echo "测试输出路径: $test_output_path"

        # 检查是否已经运行过 AF3
        if [ -f "$test_output_path/sequence_output.fasta" ] && [ -d "$test_output_path/af3_results" ]; then
            echo "检测到已存在 AF3 结果,跳过 (如需重新运行,请删除 $test_output_path/af3_results)"
            continue
        fi

        # 运行 AF3 预测
        echo "开始运行 AF3 预测..."
        step_start_time=$(date +%s)

        # 调用原始的 run_af3.sh 脚本
        bash /storage/yuanfajieLab/yuanfajie/fengyuan/Pretrain/analysis/af3/run_af3.sh "$test_output_path" 2>&1 | tee "$log_dir/${model_name}_step${step}.log"

        step_end_time=$(date +%s)
        step_cost_time=$((step_end_time - step_start_time))

        echo "完成! 耗时: ${step_cost_time}s"
    done

    echo ""
    echo "==================== 模型 $model_name 处理完成 ===================="
done

# 记录总结束时间
total_end_time=$(date +%s)
total_cost_time=$((total_end_time - total_start_time))

echo ""
echo "=========================================="
echo "所有 AF3 结构预测完成!"
echo "总耗时: ${total_cost_time}s ($(($total_cost_time / 60)) 分钟)"
echo "时间: $(date)"
echo "=========================================="
