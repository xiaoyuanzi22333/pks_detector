import torch.nn as nn
import torch
import numpy as np
from ..decoder import Simple_MLP
from .fusion import MultiModalModel


class SpatialNet(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(SpatialNet, self).__init__()
        
        self.brake_mlp_1 = Simple_MLP(in_ch, hid_ch)
        self.steer_mlp_1 = Simple_MLP(in_ch, hid_ch)
        self.throttle_mlp_1 = Simple_MLP(in_ch, hid_ch)
        
        self.brake_mlp_2 = Simple_MLP(hid_ch, out_ch)
        self.steer_mlp_2 = Simple_MLP(hid_ch, out_ch)
        self.throttle_mlp_2 = Simple_MLP(hid_ch, out_ch)
        
        self.cross_modal_attn = MultiModalModel(embed_dim=hid_ch, num_heads=1)

    def forward(self, brake, steer, throotle):
        # input [batchsize, L, 1]
        output_brake = self.brake_mlp_1(brake)
        output_brake = self.brake_mlp_2(output_brake)
        
        output_steer = self.steer_mlp_1(steer)
        output_steer = self.steer_mlp_2(output_steer)
        
        output_throttle = self.throttle_mlp_1(throotle)
        output_throttle = self.throttle_mlp_2(output_throttle)
        # output [batchsize, outch, 1]
        
        fused_1, fused_2, fused_3 = self.cross_modal_attn(output_brake, output_steer, output_throttle)
        return fused_1, fused_2, fused_3
        



