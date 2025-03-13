#!/bin/bash

# 创建输出目录（如果不存在）
mkdir -p output

run_experiment() {
    local EXP=$1
    local MAP=$2
    local TIM=$3
    
    (
        python baseline_new.py \
            --map="$MAP" \
            --num_chd=3 \
            --rand_seed=0 \
            --pth="$EXP" \
            --scl=1 \
            --time_split=$TIM \
            --time_interval=1 \
            --batch_size=32 \
            --epoch=100 \
            --scl_step=70 >> "output/output_${EXP}.txt" 2>&1
        
        echo "[$EXP] PID $! exited with code $?" >> output/status.log
    ) &
    pids+=($!)
}

run_experiment_group() {
    local group_name=$1
    local map_val=$2
    local time_val=$3
    
    echo "开始实验组: $group_name"
    pids=()
    
    run_experiment "${group_name}_1" "$map_val" "$time_val"
    run_experiment "${group_name}_2" "$map_val" "$time_val"
    run_experiment "${group_name}_3" "$map_val" "$time_val"
    run_experiment "${group_name}_4" "$map_val" "$time_val"
    run_experiment "${group_name}_5" "$map_val" "$time_val"
    
    wait "${pids[@]}"
    echo "实验组 $group_name 完成"
}

# 顺序执行不同的实验组
run_experiment_group "312_02(2)" 2 2
run_experiment_group "312_03(2)" 2 3
run_experiment_group "312_04(2)" 2 4
run_experiment_group "312_05(2)" 2 5
run_experiment_group "312_06(2)" 2 6
run_experiment_group "312_07(2)" 2 7
run_experiment_group "312_08(2)" 2 8
run_experiment_group "312_09(2)" 2 9

run_experiment_group "312_02(3)" 3 2
run_experiment_group "312_03(3)" 3 3
run_experiment_group "312_04(3)" 3 4
run_experiment_group "312_05(3)" 3 5
run_experiment_group "312_06(3)" 3 6
run_experiment_group "312_07(3)" 3 7
run_experiment_group "312_08(3)" 3 8
run_experiment_group "312_09(3)" 3 9
# 可以继续添加更多组
# run_experiment_group "312_10" 1 10

# 生成总结报告
echo "==== 实验状态报告 ===="