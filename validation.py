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
    sub_data_path = 'Data_map1_4s_1s'
    data_path = './Data_4s1s/Data_map1_4s_1s'
    data_dataset = simulator_dataset(data_path)
    train_dataset, test_dataset = generate_split_dataset(data_dataset,sub_data_path,100, False)
    model_spatial = SpatialNet_new(1,32,64,num_chd=3).cuda()
    model_temporal = TemporalNet_new_2(1, [32,32], 64, data_len=120, num_chd=3).cuda()
    model_decoder = AtNet_decoder(64,16,2).cuda()
    
    model_path = "./model_baseline/model_saved_4s_315_04-1_base5"
    dict_spatial = torch.load(model_path + "/spatial_epoch_100.pth")
    dict_temporal = torch.load(model_path + "/temporal_epoch_100.pth")
    dict_decoder = torch.load(model_path + "/decoder_epoch_100.pth")
    
    model_spatial.load_state_dict(dict_spatial)
    model_spatial.eval()
    model_temporal.load_state_dict(dict_temporal)
    model_temporal.eval()
    model_decoder.load_state_dict(dict_decoder)
    model_decoder.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化统计变量
    correct = 0          # 总正确数
    total = 0            # 总数
    correct_0 = 0        # 类别0正确数
    total_0 = 0          # 类别0总数
    correct_1 = 0        # 类别1正确数
    total_1 = 0          # 类别1总数

    with torch.no_grad():
        for batch_data in test_loader:
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
            pred_class = pred_output.argmax(dim=1)  # 获取预测类别
            label = label.long()                    # 目标标签
            
            # 计算总准确率
            correct += (pred_class == label).sum().item()
            total += batch_data[0].shape[0]

            # 计算类别0的准确率统计
            mask_0 = (label == 0)  # 找出label为0的样本
            total_0 += mask_0.sum().item()  # 类别0的总数
            correct_0 += (pred_class[mask_0] == label[mask_0]).sum().item()  # 类别0预测正确的数量

            # 计算类别1的准确率统计
            mask_1 = (label == 1)  # 找出label为1的样本
            total_1 += mask_1.sum().item()  # 类别1的总数
            correct_1 += (pred_class[mask_1] == label[mask_1]).sum().item()  # 类别1预测正确的数量

    # 计算各种准确率
    total_accuracy = correct / total if total > 0 else 0
    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

    # 打印结果
    print(f"Overall - correct: {correct}, total: {total}, accuracy: {total_accuracy:.4f}")
    print(f"Class 0 - correct: {correct_0}, total: {total_0}, accuracy: {accuracy_0:.4f}")
    print(f"Class 1 - correct: {correct_1}, total: {total_1}, accuracy: {accuracy_1:.4f}")

    return total_accuracy, accuracy_0, accuracy_1

    


if __name__ == "__main__":
    validation()
    
    # print("number: " + str(count_files("./Data_4s1s/Data_map2_4s_1s")))