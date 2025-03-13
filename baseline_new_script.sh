#!/bin/bash

# 创建输出目录（如果不存在）
mkdir -p output

run_experiment() {
    local EXP=$1
    local MAP=$2
    local TIM=$3
    
    # 在子shell中运行实验，防止变量污染
    (
        python baseline_new.py \
            --map="$MAP"\
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
# exp map time
run_experiment "312_02-3_2" 3 2
run_experiment "312_02-3_1" 3 2
run_experiment "312_02-3_3" 3 2
run_experiment "312_02-3_4" 3 2
run_experiment "312_02-3_5" 3 2


# 实时监控输出（可选）
tail -f output/output_312_09-2_{1,2,3,4,5}.txt &

# 等待所有实验完成
echo "监控中... 按 Ctrl+C 终止监控（实验会继续后台运行）"
wait "${pids[@]}"

# 生成总结报告
echo "==== 实验状态报告 ===="
cat output/status.log