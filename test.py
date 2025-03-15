from model.pks_attn_net.SpatialNet_new import SpatialNet_new
from model.pks_attn_net.TemporalNet_new import TemporalNet_new
from model.pks_attn_net.TemporalNet_new_2 import split_tensor, TemporalNet_new_2
import torch
import torch.nn as nn
import json

# # 指定 JSON 文件路径
# file_path = "./idx_folder/idx_Data_map2_5s_1s.json"

# # 打开并读取 JSON 文件
# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # 使用读取的数据

# print(len(data["train_indices"]))  
# print(len(data["test_indices"]))  

# data_all = data["train_indices"] + data["test_indices"]
# a:int = 1
# data_all = data_all[0:int(len(data_all)*a/10)]
# data_all = sorted(data_all)
# print(len(data_all))
# print(data_all[-1])
# print(len(data_all) != len(set(data_all)))

###############################

model_spat = SpatialNet_new(1,16,32)
model_temp_new = TemporalNet_new_2(1, [16,16], 32, data_len=90)
input = torch.randn(64, 90)
inputs = [input, input, input]
output_spat = model_spat(inputs)
output_temp_new = model_temp_new(inputs)
print(output_temp_new.shape)
print(output_spat.shape)

print(len(split_tensor(input)))
print(split_tensor(input)[0].shape)