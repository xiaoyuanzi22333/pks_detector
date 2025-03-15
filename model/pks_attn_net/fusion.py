import torch.nn as nn
import torch
import numpy as np
from itertools import combinations

class MultiModalModel(nn.Module):
    def __init__(self, embed_dim,num_chd, num_heads=1):
        super(MultiModalModel, self).__init__()
        self.num_chd = num_chd
        # 每对模态使用一个独立的 Cross Attention
        self.cross_attentions = nn.ModuleDict()
        for i, j in combinations(range(num_chd), 2):  # Generate all unique pairs
            self.cross_attentions[f"{i}_{j}"] = nn.MultiheadAttention(embed_dim, num_heads)
        


    def forward(self, inputs, mask=None):
        if len(inputs) != self.num_chd:
            ValueError("input dimention not fits")

        fused_outputs = [torch.zeros_like(inputs[i]) for i in range(self.num_chd)]
        # 最终融合
        for i, j in combinations(range(self.num_chd), 2):
            key = f"{i}_{j}"
            fused_ij, _ = self.cross_attentions[key](inputs[i], inputs[j], inputs[j], mask)
            fused_outputs[i] = fused_outputs[i] + fused_ij
            fused_outputs[j] = fused_outputs[j] + fused_ij

        return fused_outputs