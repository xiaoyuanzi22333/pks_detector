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
            --rand_seed=0 \
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
# exp(date_time-map_partition_expnumber) map time
run_experiment "314_04-1_10_1" 1 4 10
run_experiment "314_04-1_10_2" 1 4 10
run_experiment "314_04-1_10_3" 1 4 10

run_experiment "314_04-1_20_1" 1 4 20
run_experiment "314_04-1_20_2" 1 4 20
run_experiment "314_04-1_20_3" 1 4 20


# 实时监控输出（可选）
tail -f output/output_314_04-1_{10,20}_1.txt &

# 等待所有实验完成
echo "监控中... 按 Ctrl+C 终止监控（实验会继续后台运行）"
wait "${pids[@]}"

# 生成总结报告
echo "==== 实验状态报告 ===="
cat output/status.log