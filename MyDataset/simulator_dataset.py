import torch
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class simulator_getter(Dataset):
    # data_path = './DATA
    # src_folder = './DATA/normal_test_
    # src_file = './DATA/normal_test_/2025_01_09
    def __init__(self, data_path):
        self.brake_path = []
        self.steer_path = []
        self.throttle_path = []
        self.label = []

        
        for src_folder in os.listdir(data_path):
            
            if(src_folder.startswith('normal')):
                src_label = 0 # normal
            else:
                src_label = 1 # parkinson
            src_folder = os.path.join(data_path,src_folder)
            # print(src_folder)
            for src_file in os.listdir(src_folder):
                src_file = os.path.join(src_folder,src_file)
                # print(src_file)
                self.brake_path.append(os.path.join(src_file, 'brake_data.npy'))
                self.steer_path.append(os.path.join(src_file, 'steer_data.npy'))
                self.throttle_path.append(os.path.join(src_file, 'throttle_data.npy'))
                self.label.append(src_label)

        # brake_data = np.load(self.brake_path[0])


        

    def __getitem__(self, index):
        brake_data = np.load(self.brake_path[index])
        steer_data = np.load(self.steer_path[index])
        throttle_data = np.load(self.throttle_path[index])

        label = self.label[index]

        # datas are ndarrapy
        # label is list
        return brake_data, steer_data, throttle_data, label
    

    def __len__(self):
        return len(self.brake_path)
    


class simulator_dataset():

    def __init__(self, data_path):
        brake_sequence = []
        steer_sequence = []
        throttle_sequence = []
        self.label = []
        raw_dataset = simulator_getter(data_path)
        for brake_data, steer_data, throttle_data, label in raw_dataset:
            brake_sequence.append(torch.from_numpy(brake_data))
            steer_sequence.append(torch.from_numpy(steer_data))
            throttle_sequence.append(torch.from_numpy(throttle_data))
            self.label.append(label)
        
        self.padded_brake_seq = pad_sequence(brake_sequence, batch_first=True, padding_value=-1)
        self.padded_steer_seq = pad_sequence(steer_sequence, batch_first=True, padding_value=-1)
        self.padded_throttle_seq = pad_sequence(throttle_sequence, batch_first=True, padding_value=-1)
        self.mask = (self.padded_brake_seq != -1).float()
        
    

    def __getitem__(self, index):
        padded_brake = self.padded_brake_seq[index]
        padded_steer = self.padded_steer_seq[index]
        padded_throttle = self.padded_throttle_seq[index]
        label = self.label[index]
        mask = self.mask[index]

        return padded_brake, padded_steer, padded_throttle, mask, label
    

    def __len__(self):
        return len(self.label)









