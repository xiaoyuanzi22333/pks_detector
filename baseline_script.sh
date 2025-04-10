#!/bin/bash

# 创建输出目录（如果不存在）
mkdir -p output

run_experiment() {
    local EXP=$1
    local RAND_SEED=$2
    
    # 在子shell中运行实验，防止变量污染
    (
        python baseline_attnet.py \
            --num_chd=3 \
            --rand_seed="$RAND_SEED" \
            --pth="$EXP" \
            --scl=0 \
            --time_split=3 \
            --time_interval=1 \
            --batch_size=32 \
            --epoch=100 \
            --scl_step=70 >> "output/output_${EXP}.txt" 2>&1
        
        # 记录完成状态
        echo "[$EXP] PID $! exited with code $?" >> output/status.log
    ) &
    # 存储进程ID以便后续监控
    pids+=($!)
}

# 启动所有实验（并行执行）
run_experiment "35_02_4" 0
run_experiment "35_02_5" 1
run_experiment "35_02_6" 1

# 实时监控输出（可选）
tail -f output/output_35_01_{4,5,6}.txt &

# 等待所有实验完成
echo "监控中... 按 Ctrl+C 终止监控（实验会继续后台运行）"
wait "${pids[@]}"

# 生成总结报告
echo "==== 实验状态报告 ===="
cat output/status.log