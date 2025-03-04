@echo off

set "EXP=33_04_5"
python baseline_attnet.py ^
    "--num_chd=3"  ^
    "--rand_seed=1" ^
    "--pth=%EXP%"^
    "--scl=1" ^
    "--time_split=3" ^
    "--time_interval=1" ^
    "--batch_size=64" ^
    "--epoch=100" ^
    "--scl_step=70 " 

pause