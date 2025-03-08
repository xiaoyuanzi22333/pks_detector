import os
import numpy as np
from MyDataset.simulator_dataset import simulator_dataset
from MyDataset.split_dataset import generate_split_dataset
import utils.utils as utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.pks_attn_net.SpatialNet_new import SpatialNet_new
from model.pks_attn_net.TemporalNet_new import TemporalNet_new
from model.pks_attn_net.TemporalNet_new_2 import TemporalNet_new_2
from model.pks_attn_net.decoder import AtNet_decoder


def count_files(folder_path):
    total_files = 0
    # os.walk() 会遍历文件夹及其所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        # files 是一个列表，包含当前目录下的所有文件
        total_files += len(files)
    return total_files


def validation():
    cuda_device = 0
    batch_size = 32
    data_path = './Data_map3_3s_1s'
    test_dataset = simulator_dataset(data_path)
    model_spatial = SpatialNet_new(1,32,64).cuda()
    model_temporal = TemporalNet_new_2(1, [32,32], 64).cuda()
    model_decoder = AtNet_decoder(64,16,2).cuda()
    
    model_path = "./model_saves/model_saved_3s_38_01_1"
    dict_spatial = torch.load( model_path + "/spatial_epoch_100.pth")
    dict_temporal = torch.load( model_path + "/temporal_epoch_100.pth")
    dict_decoder = torch.load( model_path + "/decoder_epoch_100.pth")
    
    model_spatial.load_state_dict(dict_spatial)
    model_spatial.eval()
    model_temporal.load_state_dict(dict_temporal)
    model_temporal.eval()
    model_decoder.load_state_dict(dict_decoder)
    model_decoder.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    correct = 0  # 累计预测正确的样本数
    total = 0    # 累计所有有效样本数

    with torch.no_grad():
        for batch_data in test_loader:
            # if batch_data[0].shape[0] != batch_size:
            #     continue
            brake = batch_data[0].to(cuda_device).float()
            steer = batch_data[1].to(cuda_device).float()
            throttle = batch_data[2].to(cuda_device).float()
            label = batch_data[4].to(cuda_device).float()
            inputs = [brake, steer, throttle]


            # 模型前向传播
            spatial_output = model_spatial(inputs)
            temp_output = model_temporal(inputs)
            fused_output = spatial_output + temp_output
            pred_output = model_decoder(fused_output)

            # 获取预测的类别
            pred_class = pred_output.argmax(dim=1)  # 获取预测类别（取概率最大值的索引）
            label = label.long()          # 目标标签
            
            correct += (pred_class == label).sum().item()
            # print(correct)
            total += batch_data[0].shape[0]

        accuracy = correct / total
        print("correct: " + str(correct))
        print("total: " + str(total))
        return accuracy
    


if __name__ == "__main__":
    validation()
    
    # print("number: " + str(count_files("./Data_map3_3s_1s/normal")))