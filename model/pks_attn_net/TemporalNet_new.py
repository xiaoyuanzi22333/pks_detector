import torch.nn as nn
import torch
import numpy as np
from .fusion import MultiModalModel
from .SpatialNet_new import MLP_1D
from .tcn import TemporalConvNet


class TemporalNet_new(nn.Module):
    def __init__(self, in_ch: int, hid_ch: list, out_ch: int, num_chd=3):
        super(TemporalNet_new, self).__init__()
        
        self.num_chd = num_chd
        
        self.early_fusion = MLP_1D(in_ch*3, hid_ch[0])
        
        self.tcn = TemporalConvNet(hid_ch[0], hid_ch)
        
        self.mlp1 = nn.ModuleList([
            MLP_1D(hid_ch[-1], hid_ch[-1]) for _ in range(num_chd)    
        ])
        
        self.mlp2 = nn.ModuleList([
            MLP_1D(hid_ch[-1], out_ch) for _ in range(num_chd)  
        ])
        
        self.cross_modal_attn = MultiModalModel(embed_dim=out_ch, num_chd=num_chd, num_heads=1)
        self.weights = nn.Parameter(torch.ones(3) / 3)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_ch, out_ch // 2),
            nn.ReLU(),
            nn.Linear(out_ch // 2, out_ch)
        )
        
    
    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            raise ValueError("input dimension not fits")
            
        # input [batchsize, L]
        inputs_unsqueeze = []
        for i in range(self.num_chd):
            inputs_unsqueeze.append(inputs[i].unsqueeze(-1))
        
        combined = torch.cat(inputs_unsqueeze, dim=-1)
        fused_early = self.early_fusion(combined) 
        # [batchsize, L, hid_ch[0]]
        
        # 调整维度以适配 TCN
        fused_early = fused_early.permute(0, 2, 1)  # [batchsize, hid_ch[0], L]
        # TCN 处理时间序列
        tcn_output = self.tcn(fused_early)  # [batchsize, hid_ch[-1], L]
        tcn_output = tcn_output.permute(0, 2, 1)  # [batchsize, L, hid_ch[-1]]
        
        output_encode = []
        # encode TCN处理的融合变量使其具备有自己的特征
        for i in range(self.num_chd):
            output = self.mlp1[i](tcn_output) + tcn_output
            output = self.mlp2[i](output) 
            output_encode.append(output)
        
        fused_outputs = self.cross_modal_attn(output_encode)
        
        weighted_output = 0
        for i in range(self.num_chd):
            weighted_output += fused_outputs[i]*self.weights[i]
        
        # 全局池化（替代裁剪 catch）
        fused_global = torch.mean(weighted_output, dim=1)  # [batchsize, hid_ch[-1]]
        output = self.fusion_mlp(fused_global)
        
        return output
    
    
    


