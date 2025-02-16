import os
import numpy as np
import torch


#  将inputarray拆分为若干小ndarray， 
#  length为拆分出的小窗口大小
#  step为每个小窗口之间的的间隔长度
def split_ndarray(input, length=15, step=5):
    # 这个针对np.array
    num_split = (input.shape[0] - length) // step + 1
    split = [input[i:i+length] for i in range(0,num_split*step, step)]
    return np.array(split)


def split_tensor(input_tensor, length=30, step=15):
    # input_tensor 的形状为 (batch_size, width)
    batch_size, width = input_tensor.size()
    # 计算切片数量
    num_slices = (width - length) // step + 1
    # 创建一个列表来存储切片
    slices = []
    # 提取切片
    for i in range(num_slices):
        start_index = i * step
        end_index = start_index + length
        slices.append(input_tensor[:, start_index:end_index])
    # 将切片列表转换为 Tensor
    slices_tensor = torch.stack(slices)
    slices_tensor = slices_tensor.permute(1,0,2)
    return slices_tensor  # 输出形状为 (num_slices, batch_size, length)


if __name__ == "__main__":
        # 示例
    batch_size = 4
    width = 100
    input_tensor = torch.randn(batch_size, width)

    # 调用函数
    output_tensor = split_tensor(input_tensor)

    print("输出形状:", output_tensor.shape)  # 应输出 (num_slices, batch_size, length)