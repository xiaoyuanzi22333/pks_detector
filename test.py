from model.pks_attn_net.SpatialNet_new import SpatialNet_new
from model.pks_attn_net.TemporalNet_new import TemporalNet_new
import torch
import torch.nn as nn


model_spat = SpatialNet_new(1,16,32)
model_temp = TemporalNet_new(1, [16,16], 32)
input = torch.randn(64, 90)
inputs = [input, input, input]
output_spat = model_spat(inputs)
output_temp = model_temp(inputs)

print(output_spat.shape)
print(output_temp.shape)