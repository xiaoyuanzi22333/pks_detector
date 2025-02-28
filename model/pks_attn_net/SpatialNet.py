import torch.nn as nn
import torch
import numpy as np
from ..decoder_old import Simple_MLP
from .fusion import MultiModalModel


class SpatialNet(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_chd):
        super(SpatialNet, self).__init__()
        self.num_chd = num_chd
        self.mlp_1 = nn.ModuleList([
            Simple_MLP(in_ch, hid_ch) for _ in range(num_chd)
        ])
        
        self.mlp_2 = nn.ModuleList([
            Simple_MLP(hid_ch, out_ch) for _ in range(num_chd)
        ])
        
        self.cross_modal_attn = MultiModalModel(embed_dim=out_ch,num_chd=num_chd ,num_heads=1)

    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            ValueError("input dimention not fits")
            
        # input [batchsize, L, 1]
        output_chds = []
        for i in range(self.num_chd):
            output = self.mlp_1[i](inputs[i])
            output = self.mlp_2[i](output)
            output_chds.append(output)
        # output [batchsize, outch, 1]

        fused_output = self.cross_modal_attn(output_chds)
        output = sum(fused_output)
        # [batch_size, L, out_ch]

        return output
        



