import torch.nn as nn
import torch
import numpy as np
from .fusion import MultiModalModel
from .SpatialNet_new import MLP_1D
from .tcn import TemporalConvNet




class TemporalNet_new_2(nn.Module):
    def __init__(self, in_ch: int, hid_ch: list, out_ch: int, data_len, win_len=30, step=15, num_chd=3):
        super(TemporalNet_new_2, self).__init__()
        self.num_chd = num_chd
        self.split_len = (data_len - win_len) // 15 + 1
        self.win_len = win_len
        self.Head_tcns = nn.ModuleList([
            Head_fusion(in_ch,hid_ch[0],num_chd) for _ in range(self.split_len)
        ])
        
        self.tcn = TemporalConvNet(hid_ch[0], hid_ch)
        self.mlp1 = MLP_1D(hid_ch[-1], hid_ch[-1])
        self.mlp2 = MLP_1D(hid_ch[-1], out_ch)
        self.out_ch = out_ch
        
        self.cross_modal_attn = MultiModalModel(embed_dim=out_ch, num_chd=self.split_len, num_heads=1)
        self.weights = nn.Parameter(torch.ones(self.split_len) / self.split_len)
    

    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            raise ValueError("input dimension not fits")
        
        # input [[batch_size, L] * num_chd]
        # 对每个模态的数据进行分割
        heads_input = []
        for i in range(self.num_chd):
            split_inputs = split_tensor(inputs[i]) # [ [batch_size, win_len] * split_len ]
            if len(split_inputs) != self.split_len:
                raise ValueError("check the splited windows")
            heads_input.append(split_inputs) # [[ [batch_size, win_len] * split_len ] * num_chd]
        
        heads_input = [[heads_input[j][i] for j in range(self.num_chd)] for i in range(self.split_len)]
        # [[ [batch_size, win_len] * num_chd ] * split_len]
        
        # 切割之后的数据依次放入head_tcn进行处理
        heads_output = []
        for j in range(self.split_len):
            head_output = self.Head_tcns[j](heads_input[j]) # [batch_size, win_len, hid_ch[0]]
            # print("head_output: " + str(head_output.shape))
            heads_output.append(head_output) # [[batch_size, win_len, hid_ch[0]] * split_len]
        tcn_input = torch.cat(heads_output, dim=1) # [batch_size, win_len*split_len, hid_ch[0]]
        
        tcn_input = tcn_input.permute(0,2,1)
        tcn_output = self.tcn(tcn_input)
        tcn_output = tcn_output.permute(0,2,1)
            
        # encode TCN处理的融合变量使其具备有自己的特征
        output = self.mlp1(tcn_output) + tcn_output
        output = self.mlp2(output)
        output_encode = torch.split(output, self.win_len, dim=1)

        # 使用attetion
        fused_outputs = self.cross_modal_attn(output_encode)
        # 不使用attention
        # fused_outputs = output_encode
        
        weighted_output = 0
        for i in range(self.split_len):
            weighted_output += fused_outputs[i]*self.weights[i]
        
        # 全局池化（替代裁剪 catch）
        fused_global = torch.mean(weighted_output, dim=1)  # [batchsize, hid_ch[-1]]
        
        return fused_global


class Head_fusion(nn.Module):
    def __init__(self, in_ch, hid_ch, num_chd):
        super(Head_fusion, self).__init__()
        self.num_chd = num_chd      
        self.early_fusion = MLP_1D(in_ch*num_chd, hid_ch)     
        
    
    def forward(self, inputs):
        if len(inputs) != self.num_chd:
            raise ValueError("input dimension not fits")
        
        # inputs [[batchsize, L] * num_chd]
        inputs_unsqueeze = []
        for i in range(self.num_chd):
            inputs_unsqueeze.append(inputs[i].unsqueeze(-1))
        
        combined = torch.cat(inputs_unsqueeze, dim=-1) # [batchsize, L, num_chd]
        # print("combined: " + str(combined.shape))
        fused_early = self.early_fusion(combined) 
        # [batchsize, L, hid_ch[0]]
        
        return fused_early
    



def split_tensor(tensor, win_len=30, step=15):
    batch_size, L = tensor.shape
    if L < win_len:
        raise ValueError("输入序列长度 L 必须大于等于 win_len")
    
    # 计算可以切割的窗口数量
    num_windows = (L - win_len) // step + 1
    
    # 存储结果的列表
    result = []
    
    # 手动切片
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + win_len
        window = tensor[:, start_idx:end_idx]
        result.append(window)
    
    return result