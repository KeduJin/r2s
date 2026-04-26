#!/bin/bash
# 计算 sacct 输出中所有任务的总时长
# 用法: ./tmp_sacct_total_elapsed.sh
# 或指定日期范围: ./tmp_sacct_total_elapsed.sh 2025-11-19 2026-01-30

START_DATE="${1:-2025-11-19}"
END_DATE="${2:-2026-01-30}"

# 将 Elapsed 格式转为秒数
# 支持: HH:MM:SS 或 D-HH:MM:SS
elapsed_to_seconds() {
    local s="$1"
    [[ -z "$s" || "$s" == "Elapsed" ]] && echo 0 && return
    local days=0
    if [[ "$s" == *-* ]]; then
        days="${s%%-*}"
        s="${s#*-}"
    fi
    IFS=':' read -r h m sec <<< "$s"
    h=${h:-0}; m=${m:-0}; sec=${sec:-0}
    echo $((10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$sec))
}

# 秒数转可读格式
seconds_to_readable() {
    local total=$1
    local days=$((total / 86400))
    local rem=$((total % 86400))
    local hours=$((rem / 3600))
    rem=$((rem % 3600))
    local mins=$((rem / 60))
    local secs=$((rem % 60))
    if [[ $days -gt 0 ]]; then
        printf "%d天 %02d:%02d:%02d\n" "$days" "$hours" "$mins" "$secs"
    else
        printf "%02d:%02d:%02d\n" "$hours" "$mins" "$secs"
    fi
}

total_sec=0
count=0

# 跳过表头，按 | 分割取第3列 Elapsed
while IFS='|' read -r _ _ elapsed _; do
    [[ -z "$elapsed" ]] && continue
    sec=$(elapsed_to_seconds "$elapsed")
    total_sec=$((total_sec + sec))
    count=$((count + 1))
done < <(sacct -u "$USER" --partition=public-h800,public-h100 -S "$START_DATE" -E "$END_DATE" -X -o JobID,Partition,Elapsed,AllocTRES -P 2>/dev/null | tail -n +3)

total_hours=$(echo "scale=2; $total_sec / 3600" | bc)
cost=$(echo "scale=2; $total_hours * 40" | bc)

echo "日期范围: $START_DATE ~ $END_DATE"
echo "任务数量: $count"
echo "总时长(秒): $total_sec"
echo "总时长(小时): $total_hours"
echo "总时长(天): $(seconds_to_readable $total_sec)"
echo "费用(小时*40): $cost"
