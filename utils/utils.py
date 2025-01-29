import os
import numpy as np
import torch


#  将inputarray拆分为若干小ndarray， 
#  length为拆分出的小窗口大小
#  step为每个小窗口之间的的间隔长度
def split_ndarray(input, length=15, step=5):
    num_split = (input.shape[0] - length) // step + 1
    split = [input[i:i+length] for i in range(0,num_split*step, step)]
    return np.array(split)


def split_tensor(input_tensor, length=15, step=5):
    # 计算可以提取的切片数量
    num_slices = (input_tensor.size(2) - length) // step + 1

    # 创建一个列表来存储切片
    slices = []

    # 提取切片
    for i in range(num_slices):
        start_index = i * step
        end_index = start_index + length
        slices.append(input_tensor[:, :, start_index:end_index])

    # 将切片列表转换为 Tensor
    slices_tensor = torch.stack(slices)
    slices_tensor = slices_tensor.reshape(slices_tensor.shape[0],slices_tensor.shape[1],-1)

    return slices_tensor


def train_mask():
    # 计算损失时，使用掩码
    losses = criterion(outputs.view(-1, num_classes), targets.view(-1))  # 计算所有位置的损失
    masked_losses = losses * mask.view(-1)  # 仅保留有效位置的损失
    total_loss = masked_losses.sum() / mask.sum()  # 平均损失