import torch.nn as nn
import torch
import numpy as np
from .tcn import TemporalConvNet
from .fusion import MultiModalModel
from utils.utils import split_tensor
from ..decoder import Simple_MLP


class TemporalNet(nn.Module):
    def __init__(self, in_ch: int, hid_ch: list, out_ch: int, catch: int):
        super(TemporalNet, self).__init__()
        self.catch = catch
        self.tcn = TemporalConvNet(in_ch, hid_ch)
        self.cross_modal_attn = MultiModalModel(embed_dim=hid_ch[-1], num_heads=1)
        self.mlp = Simple_MLP(hid_ch[-1]*catch, out_ch)

    def forward(self, brake, steer, throotle):
        # input [batchsize, L]
        
        # 切割数据以放入tcn
        brake = split_tensor(brake, 30, 15)
        throotle = split_tensor(throotle, 30, 15)
        steer = split_tensor(steer, 30, 15)
        # output [batchsize, seq_len(10), 30]
        
        # 再次切换维度
        brake = brake.permute(0, 2, 1)
        steer = steer.permute(0, 2, 1)
        throotle = throotle.permute(0, 2, 1)
        # print("steer shape")
        # print(steer.shape)
        # output [batchsize, 30, seq_len(10)]
        
        # 送入tcn
        output_brake = self.tcn(brake)
        output_steer = self.tcn(steer)
        output_throttle = self.tcn(throotle)
        # output [batchsize, hid_ch[-1], seq_len(10)]
        

        output_brake = output_brake.permute(0,2,1)[:,-self.catch:,:]
        output_steer = output_steer.permute(0,2,1)[:,-self.catch:,:]
        output_throttle = output_throttle.permute(0,2,1)[:,-self.catch:,:]
        # print( "output steer shape: " + str(output_brake.shape))
        
        # output [batchsize, 5, hid_ch[-1]]
        # 输入交叉注意力模型
        fused_1, fused_2, fused_3 = self.cross_modal_attn(output_brake, output_steer, output_throttle)
        # fused_n [batchsize, 5. hid_ch[-1]]
        fused_1 = fused_1.view(fused_1.shape[0],-1)
        fused_2 = fused_2.view(fused_2.shape[0],-1)
        fused_3 = fused_3.view(fused_3.shape[0],-1)
        fused = (fused_1+fused_2+fused_3)

        output = self.mlp(fused)

        return output
    
