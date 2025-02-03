import os
import numpy as np
import matplotlib.pyplot as plt

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
        simulated_steer = src_steer + 0.01 * filtered_noise  # 设置抖动的振幅
        simulated_throttle = src_throttle + 0.01 * filtered_noise
        simulated_throttle = [0 if i < 0 else i for i in simulated_throttle]
        simulated_brake = src_brake
        
        np.save(tgt + '/brake_data.npy', simulated_brake)
        np.save(tgt + '/steer_data.npy', simulated_steer)
        np.save(tgt + '/throttle_data.npy', simulated_throttle)
        


def visulize_noise_data(t, src, simulated):
    # 绘制原始方向盘数据
    plt.figure(figsize=(12, 6))
    plt.plot(t, src, label="Normal Steering Data", alpha=0.7)

    # 绘制加入帕金森抖动后的信号
    plt.plot(t, simulated, label="With Parkinson Tremor Noise", alpha=0.7)

    # 添加图例和标题
    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle")
    plt.title("Steering Data with Parkinson Tremor Noise")
    plt.legend()
    plt.show()





if __name__ == "__main__":
    src_folder = "./Data/normal_test_left"
    tgt_folder = "./Data/abnormal_test_right_gen"
    data_add_noise(src_folder, tgt_folder)