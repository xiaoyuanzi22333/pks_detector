import torch.nn as nn
import torch
import numpy as np
from .tcn import TemporalConvNet
from .fusion import MultiModalModel
from utils.utils import split_tensor
from ..decoder_old import Simple_MLP


class TemporalNet(nn.Module):
    def __init__(self, in_ch: int, hid_ch: list, out_ch: int, catch: int, num_chd: int):
        super(TemporalNet, self).__init__()
        self.num_chd = num_chd
        self.catch = catch
        self.tcn_list = nn.ModuleList([
            TemporalConvNet(in_ch, hid_ch) for _ in range(num_chd)
        ])
        # self.tcn = TemporalConvNet(in_ch, hid_ch)
        self.cross_modal_attn = MultiModalModel(embed_dim=hid_ch[-1],num_chd=num_chd, num_heads=1)
        self.mlp = Simple_MLP(hid_ch[-1]*catch, out_ch)

    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            ValueError("input dimention not fits")
        # input [batchsize, L]
        
        outputs = []
        
        for i in range(self.num_chd):
            # 切割数据以放入tcn
            splited_out = split_tensor(inputs[i], 30, 15)
            # output [batchsize, seq_len(10), 30]
            # 再次切换维度
            splited_out = splited_out.permute(0, 2, 1)
            # 送入TCN
            splited_out = self.tcn_list[i](splited_out)
            # splited_out = self.tcn(splited_out)
            # output [batchsize, hid_ch[-1], seq_len(10)]
            # catch后面末尾的输出
            splited_out = splited_out.permute(0,2,1)[:,-self.catch:,:]
            outputs.append(splited_out)
        
        # output [batchsize, 5, hid_ch[-1]]
        # 输入交叉注意力模型
        fused_outputs = self.cross_modal_attn(outputs)
        
        for i in range(self.num_chd):
            # fused_n [batchsize, 5. hid_ch[-1]]
            fused_outputs[i] = fused_outputs.view(fused_outputs[i].shape[0],-1)
            
        fused = sum(fused_outputs)
        output = self.mlp(fused)

        return output
    
