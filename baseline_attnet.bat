@echo off

set "EXP=35_01_1"
python baseline_attnet.py ^
    --num_chd=3  ^
    --rand_seed=0 ^
    --pth=%EXP% ^
    --scl=1 ^
    --time_split=3 ^
    --time_interval=1 ^
    --batch_size=32 ^
    --epoch=100 ^
    --scl_step=70 ^ 
pause