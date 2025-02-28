@echo off

set "EXP=0228_01"
python baseline_attnet.py ^
    --pth=%EXP% ^
    --scl=1 ^
    --time_split=3 ^
    --time_interval=1 ^
    --batch_size=32 ^
    --epoch=90 ^
    --scl_step=60 >> output/output_%EXP%.txt 2>&1

pause