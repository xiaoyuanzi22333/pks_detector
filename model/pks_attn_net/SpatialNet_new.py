import torch.nn as nn
import torch
import numpy as np
from .fusion import MultiModalModel


class SpatialNet_new(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_chd=3):
        super(SpatialNet_new, self).__init__()
        
        self.num_chd = num_chd
        
        self.early_fusion = MLP_1D(in_ch*num_chd, hid_ch)

        self.mlp_1 = nn.ModuleList([
            MLP_1D(hid_ch, hid_ch) for _ in range(num_chd)
        ])
        
        self.mlp_2 = nn.ModuleList([
            MLP_1D(hid_ch, out_ch) for _ in range(num_chd)
        ])
        
        self.cross_modal_attn = MultiModalModel(embed_dim=out_ch,num_chd=num_chd ,num_heads=1)
        self.weights = nn.Parameter(torch.ones(num_chd) / num_chd)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_ch, out_ch // 2),
            nn.ReLU(),
            nn.Linear(out_ch // 2, out_ch)
        )
        
    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            raise ValueError("input dimension not fits")
            
        # input [batchsize, L]
        input_unsqueeze = []
        for i in range(self.num_chd):
            input_unsqueeze.append(inputs[i].unsqueeze(-1))

        # [batchsize, L, 1]
        combined = torch.cat(input_unsqueeze, dim=-1)
        # [batchsize, L, 3]
        fused_early = self.early_fusion(combined)
        # [batchsize, L, hid_ch]

        output_chds = []
        for i in range(self.num_chd):
            output = self.mlp_1[i](fused_early) + fused_early
            output = self.mlp_2[i](output) 
            output_chds.append(output)
        # output [batchsize, L, out_ch]

        fused_outputs = self.cross_modal_attn(output_chds)
        
        output_weight = 0
        for i in range(self.num_chd):
            output_weight += fused_outputs[i]*self.weights[i]
        # [batch_size, L, out_ch]
        fused_global = torch.mean(output_weight, dim=1)
        # [batch_size, out_ch]
        output = self.fusion_mlp(fused_global)
        
        return output
        
    


class MLP_1D(nn.Module):
    def __init__(self, in_ch, out_ch): 
        super(MLP_1D, self).__init__()

        self.ln = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.ru = nn.LeakyReLU(0.1)

    
    def forward(self, input):
        output = self.ln(input)
        output = output.transpose(1,2)
        output = self.bn(output)
        output = output.transpose(1,2)
        output = self.ru(output)

        return output