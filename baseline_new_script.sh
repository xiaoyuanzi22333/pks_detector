#!/bin/bash

# 创建输出目录（如果不存在）
mkdir -p output

run_experiment() {
    local EXP=$1
    local MAP=$2
    local TIM=$3
    local PAR=$4
    
    # 在子shell中运行实验，防止变量污染
    (
        python baseline_new.py \
            --map="$MAP"\
            --partition="$PAR"\
            --num_chd=3 \
            --rand_seed=1 \
            --pth="$EXP" \
            --scl=1 \
            --time_split=$TIM \
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
# exp(date_time-map_partition_expnumber) map time partition
run_experiment "317_04-1_base1" 1 4 100
run_experiment "317_04-1_base2" 1 4 100
run_experiment "317_04-1_base3" 1 4 100
run_experiment "317_04-1_base4" 1 4 100
run_experiment "317_04-1_base5" 1 4 100

run_experiment "317_04-2_base1" 2 4 100
run_experiment "317_04-2_base2" 2 4 100
run_experiment "317_04-2_base3" 2 4 100
run_experiment "317_04-2_base4" 2 4 100
run_experiment "317_04-2_base5" 2 4 100


run_experiment "317_04-3_base1" 3 4 100
run_experiment "317_04-3_base2" 3 4 100
run_experiment "317_04-3_base3" 3 4 100
run_experiment "317_04-3_base4" 3 4 100
run_experiment "317_04-3_base5" 3 4 100


# 实时监控输出（可选）
tail -f output/output_317_04-1_base1.txt &

# 等待所有实验完成
echo "监控中... 按 Ctrl+C 终止监控（实验会继续后台运行）"
wait "${pids[@]}"

