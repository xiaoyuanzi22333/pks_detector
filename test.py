from model.pks_attn_net.SpatialNet_new import SpatialNet_new
from model.pks_attn_net.TemporalNet_new import TemporalNet_new
from model.pks_attn_net.TemporalNet_new_2 import split_tensor, TemporalNet_new_2
import torch
import torch.nn as nn


# model_spat = SpatialNet_new(1,16,32)
# model_temp = TemporalNet_new(1, [16,16], 32)
model_temp_new = TemporalNet_new_2(1, [16,16], 32)
input = torch.randn(64, 90)
inputs = [input, input, input]
# output_spat = model_spat(inputs)
# output_temp = model_temp(inputs)
output_temp_new = model_temp_new(inputs)
print(output_temp_new.shape)

# print(output_spat.shape)
# print(output_temp.shape)

# print(len(split_tensor(input)))
# print(split_tensor(input)[0].shape)