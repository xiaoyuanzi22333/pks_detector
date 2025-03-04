import torch.nn as nn
import torch
import numpy as np
from ..decoder_old import Simple_MLP
from ..encoder_old import ResBasicBlock
from .fusion import MultiModalModel


class SpatialNet(nn.Module):
    def __init__(self, in_ch, hid_ch, num_chd):
        super(SpatialNet, self).__init__()
        self.num_chd = num_chd
        

        self.mlp_1 = nn.ModuleList([
            Simple_MLP(in_ch, hid_ch) for _ in range(num_chd)
        ])
        
        self.mlp_2 = nn.ModuleList([
            Simple_MLP(hid_ch, in_ch) for _ in range(num_chd)
        ])
        
        self.cross_modal_attn = MultiModalModel(embed_dim=in_ch,num_chd=num_chd ,num_heads=1)

    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            ValueError("input dimention not fits")
        
        # print("input")
        # print(inputs[0].shape)
            
        # input [batchsize, L]
        output_chds = []
        for i in range(self.num_chd):
            output = self.mlp_1[i](inputs[i])
            output = self.mlp_2[i](output)
            output_chds.append(output)
        # output [batchsize, outch]

        fused_output = self.cross_modal_attn(output_chds)
        output = sum(fused_output)
        # print("output: ")
        # print(output.shape)
        # exit()
        # [batch_size, out_ch]

        return output
        
    

class SpatialNet_Res(nn.Module):
    def __init__(self, in_ch, hid_ch, num_chd):
        super(SpatialNet_Res, self).__init__()
        self.num_chd = num_chd

        self.cbw = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, hid_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(hid_ch),
                nn.LeakyReLU(0.1),
            ) for _ in range(num_chd)
        ])

        self.resnet_1 = nn.ModuleList([
            ResBasicBlock(in_ch=hid_ch, hid_ch=hid_ch, out_ch=hid_ch) 
            for _ in range(num_chd)
        ])

        self.resnet_2 = nn.ModuleList([
            ResBasicBlock(in_ch=hid_ch, hid_ch=hid_ch, out_ch=hid_ch) 
            for _ in range(num_chd)
        ])

        self.cross_modal_attn = MultiModalModel(embed_dim=hid_ch, num_chd=num_chd, num_heads=1)

        self.btn = nn.BatchNorm1d(hid_ch)

    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            raise ValueError("Input dimension does not match num_chd")
            
        # 输入格式假设为 [batch_size, L, in_ch]，需要转为 [batch_size, in_ch, L]
        output_chds = []
        # 调整输入维度
        for i in range(self.num_chd):
            # print("input: " + str(inputs[i].shape))
            x = inputs[i].unsqueeze(1)  # [batch_size, in_ch, L]
            # print("X: " + str(x.shape))
            output = self.cbw[i](x)
            output = self.resnet_1[i](output)   # [batch_size, hid_ch, L]
            output = self.resnet_2[i](output)  # [batch_size, out_ch, L]
            output = output.transpose(1,2)
            # print("output: " + str(output.shape))
            # 转回 [batch_size, L, out_ch] 以匹配后续操作
            # [batch_size, L, out_ch]
            output_chds.append(output)

        # 注意力融合
        fused_output = self.cross_modal_attn(output_chds)
        fused_output = sum(fused_output)    # [batch_size, L, out_ch]
        pooled_fused = fused_output.mean(dim=2, keepdim=True)
        # print("pooled_fused: " + str(pooled_fused.shape))
        pooled_fused = self.btn(pooled_fused)
        pooled_fused = pooled_fused.squeeze(2) 
        # print("pooled_fused: " + str(pooled_fused.shape))
        # exit()

        return pooled_fused