import torch.nn as nn
import torch
import numpy as np
from .tcn import TemporalConvNet
from .fusion import MultiModalModel
from utils.utils import split_tensor

class TemporalNet(nn.Module):
    def __init__(self, in_ch: int, hid_ch: list, out_ch: int):
        super(TemporalNet, self).__init__()
        self.tcn = TemporalConvNet(in_ch, hid_ch, out_ch)
        self.cross_modal_attn = MultiModalModel(embed_dim=hid_ch[-1], num_heads=1)
        

    def forward(self, brake, steer, throotle):
        # input [batchsize, L, 1]
        # 先换维度
        brake = brake.permute(0, 2, 1)
        steer = steer.permute(0, 2, 1)
        throotle = throotle.permute(0, 2, 1)
        # output [batchsize, 1, L]
        
        # 切割数据以放入tcn
        brake = split_tensor(brake, 30, 10)
        throotle = split_tensor(throotle, 30, 10)
        steer = split_tensor(steer, 30, 10)
        # output [batchsize, seq_len(10), 30]
        
        # 再次切换维度
        brake = brake.permute(0, 2, 1)
        steer = steer.permute(0, 2, 1)
        throotle = throotle.permute(0, 2, 1)
        # output [batchsize, 30, seq_len(10)]
        
        # 送入tcn
        output_brake = self.tcn(brake)
        output_steer = self.tcn(steer)
        output_throttle = self.tcn(throotle)
        # output [batchsize, outch, seq_len(10)]
        
        fused_1, fused_2, fused_3 = self.cross_modal_attn(output_brake, output_steer, output_throttle)
        # fused_n [batchsize, outch, seq_len(10)]
        return fused_1, fused_2, fused_3
    
