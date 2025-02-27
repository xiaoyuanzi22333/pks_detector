@echo off
python baseline_attnet.py ^
    --pth=0227_04 ^
    --scl=0 ^
    --time_split=3 ^
    --time_interval=1 ^
    --batch_size=32 ^
    --epoch=90 ^
    --scl_step=60

pause