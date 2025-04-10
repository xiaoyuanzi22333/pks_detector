import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import time


def data_reverse(src_folders, tgt_folders):
    if not os.path.exists(tgt_folders):
        os.mkdir(tgt_folders)

    for folder in os.listdir(src_folders):
        src = os.path.join(src_folder, folder)
        tgt = os.path.join(tgt_folder, folder)
        if not os.path.exists(tgt):
            os.mkdir(tgt)
        src_brake = np.load(src + '/brake_data.npy')
        src_steer = np.load(src + '/steer_data.npy')
        src_throttle = np.load(src + '/throttle_data.npy')
        
        tgt_brake = src_brake
        tgt_steer = src_steer * -1
        tgt_throttle = src_throttle
        
        np.save(tgt + '/brake_data.npy', tgt_brake)
        np.save(tgt + '/steer_data.npy', tgt_steer)
        np.save(tgt + '/throttle_data.npy', tgt_throttle)




def data_add_noise(src_folder, tgt_folder):
    fs = 30
    if not os.path.exists(tgt_folder):
        os.mkdir(tgt_folder)
    
    for folder in os.listdir(src_folder):
        src = os.path.join(src_folder, folder)
        tgt = os.path.join(tgt_folder, folder)
        if not os.path.exists(tgt):
            os.mkdir(tgt)
        src_brake = np.load(src + '/brake_data.npy')
        src_steer = np.load(src + '/steer_data.npy')
        src_throttle = np.load(src + '/throttle_data.npy')
        
        # 设置参数
        t = np.linspace(0, len(src_steer)/fs, len(src_steer))
        freq = 5  #频率
        
        white_noise = np.random.normal(size=len(t))
        # FFT 变换
        f = np.fft.fftfreq(len(t), d=1/fs)  # 计算频率轴
        fft_noise = np.fft.fft(white_noise)  # 对噪声信号进行 FFT

        # 设计带通滤波器 (4-6 Hz)
        bandpass_filter = (np.abs(f) >= 4) & (np.abs(f) <= 6)
        filtered_fft_noise = fft_noise * bandpass_filter  # 应用带通滤波器

        # 逆 FFT 变换回时域
        filtered_noise = np.fft.ifft(filtered_fft_noise).real  # 取实部，得到带通噪声
        simulated_steer = src_steer + 0.1 * filtered_noise  # 设置抖动的振幅
        simulated_throttle = src_throttle + 0.1 * filtered_noise
        simulated_throttle = [0 if i < 0 else i for i in simulated_throttle]
        simulated_brake = src_brake
        
        np.save(tgt + '/brake_data.npy', simulated_brake)
        np.save(tgt + '/steer_data.npy', simulated_steer)
        np.save(tgt + '/throttle_data.npy', simulated_throttle)
        


def visulize_noise_data(src, simulated):
    src = src + "/steer_data.npy"
    simulated = simulated + "/steer_data.npy"
    src = np.load(src)
    simulated = np.load(simulated)
    t = np.linspace(0,10, len(src))
    print(len(t))
    print(len(src))
    
    # 绘制原始方向盘数据
    plt.figure(figsize=(12, 6))
    plt.plot(t, src, label="Normal Steering Data", alpha=0.7)

    # 绘制加入帕金森抖动后的信号
    plt.plot(t, simulated, label="With Parkinson Tremor Noise", alpha=0.7)

    # 添加图例和标题
    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle")
    plt.title("Steering Data with Parkinson Tremor Noise")
    save_path = f'plots/steering_plot_{int(time.time())}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()



def split_data_seconds(src, target, split_time, interval_time):
    if not os.path.exists(target):
        os.mkdir(target)
    for folder_class in os.listdir(src):
        tgt_folder = os.path.join(target, folder_class)
        if not os.path.exists(tgt_folder):
            os.mkdir(tgt_folder)
        for sample in os.listdir(os.path.join(src, folder_class)):
            src_file = os.path.join(src, folder_class, sample)
            tgt_file = os.path.join(target, folder_class, sample)
            brake_data = np.load(os.path.join(src_file, 'brake_data.npy'))
            steer_data = np.load(os.path.join(src_file, 'steer_data.npy'))
            throttle_data = np.load(os.path.join(src_file, 'throttle_data.npy'))
            fs = 30
            for i in range(0, len(brake_data), fs*interval_time):
                if not os.path.exists(tgt_file + str(i)):
                    os.mkdir(tgt_file + str(i))
                np.save(os.path.join(tgt_file + str(i), 'brake_data.npy'), brake_data[i:i+fs*split_time])
                np.save(os.path.join(tgt_file + str(i), 'steer_data.npy'), steer_data[i:i+fs*split_time])
                np.save(os.path.join(tgt_file + str(i), 'throttle_data.npy'), throttle_data[i:i+fs*split_time])
                


def fuse_dataset_together(src_folder, tgt_folder):
    shutil.copytree(src_folder, tgt_folder, dirs_exist_ok=True)
    


if __name__ == "__main__":
    # src_folder = "./Data/normal_test_left"
    # tgt_folder = "./Data/abnormal_test_left"
    # data_add_noise(src_folder, tgt_folder)
    # src_folder = "./Data/normal_test_right"
    # tgt_folder = "./Data/abnormal_test_right"
    # data_add_noise(src_folder, tgt_folder)
    # src_folder = "./Data/normal_test_left_gen"
    # tgt_folder = "./Data/abnormal_test_left_gen"
    # data_add_noise(src_folder, tgt_folder)
    # src_folder = "./Data/normal_test_right_gen"
    # tgt_folder = "./Data/abnormal_test_right_gen"
    # data_add_noise(src_folder, tgt_folder)
    
    # src_folder = "./Data_3s_1s/normal_test_left/2025_01_09_19-38-30120"
    # tgt_folder = "./Data_3s_1s/abnormal_test_left/2025_01_09_19-38-30120"
    # visulize_noise_data(src_folder, tgt_folder)
    
    # src_folder1 = "./Data_map1_3s_1s/abnormal"
    # src_folder2 = "./Data_map2_3s_1s/abnormal"
    # src_folder3 = "./Data_map3_3s_1s/abnormal"
    # tgt_folder = "./Data_map0_3s_1s/abnormal"
    # fuse_dataset_together(src_folder1, tgt_folder)
    # fuse_dataset_together(src_folder2, tgt_folder)
    # fuse_dataset_together(src_folder3, tgt_folder)
    
    time_interval = 1
    for j in range(5,9):
        time_split = j+2
        for i in range(3):
            map_num = i+1
            src_folder = './Data_new_large_311/map' + str(map_num)
            if not os.path.exists('./Data_'+str(time_split)+'s'+str(time_interval)+'s'):
                os.mkdir('./Data_'+str(time_split)+'s'+str(time_interval)+'s')
            tgt_folder = './Data_'+str(time_split)+'s'+str(time_interval)+'s/Data_map'+str(map_num)+'_'+str(time_split)+'s_'+str(time_interval)+'s'
            print("time split: " + str(time_split) + ", map number: " + str(map_num))
            split_data_seconds(src_folder, tgt_folder, time_split, time_interval)