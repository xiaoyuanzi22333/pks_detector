import torch.nn as nn
import torch
import numpy as np
from ..decoder_old import Simple_MLP
from .fusion import MultiModalModel


class SpatialNet(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(SpatialNet, self).__init__()
        
        self.chd1_mlp_1 = Simple_MLP(in_ch, hid_ch)
        self.chd2_mlp_1 = Simple_MLP(in_ch, hid_ch)
        self.chd3_mlp_1 = Simple_MLP(in_ch, hid_ch)
        
        self.chd1_mlp_2 = Simple_MLP(hid_ch, out_ch)
        self.chd2_mlp_2 = Simple_MLP(hid_ch, out_ch)
        self.chd3_mlp_2 = Simple_MLP(hid_ch, out_ch)
        
        self.cross_modal_attn = MultiModalModel(embed_dim=out_ch, num_heads=1)

    def forward(self, chd1, chd2, throotle):
        # input [batchsize, L, 1]
        output_chd1 = self.chd1_mlp_1(chd1)
        output_chd1 = self.chd1_mlp_2(output_chd1)
        
        output_chd2 = self.chd2_mlp_1(chd2)
        output_chd2 = self.chd2_mlp_2(output_chd2)
        
        output_chd3 = self.chd3_mlp_1(throotle)
        output_chd3 = self.chd3_mlp_2(output_chd3)
        # output [batchsize, outch, 1]

        fused_1, fused_2, fused_3 = self.cross_modal_attn(output_chd1, output_chd2, output_chd3)
        output = (fused_1+fused_2+fused_3)

        return output
        



