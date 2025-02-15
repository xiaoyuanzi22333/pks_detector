import torch.nn as nn
import torch
import numpy as np


class MultiModalModel(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(MultiModalModel, self).__init__()
        # 每对模态使用一个独立的 Cross Attention
        self.cross_attention_12 = nn.MultiheadAttention(embed_dim, num_heads)  # 模态 1 和 2
        self.cross_attention_13 = nn.MultiheadAttention(embed_dim, num_heads)  # 模态 1 和 3
        self.cross_attention_23 = nn.MultiheadAttention(embed_dim, num_heads)  # 模态 2 和 3

    def forward(self, brake, steer, throttle, mask=None):
        fused_12, _ = self.cross_attention_12(brake, steer, steer, mask)
        fused_13, _ = self.cross_attention_13(brake, throttle, throttle, mask)
        fused_23, _ = self.cross_attention_23(steer, throttle, throttle, mask)

        # 最终融合
        fused_1 = fused_12 + fused_13  # 模态 1 的最终融合
        fused_2 = fused_12 + fused_23  # 模态 2 的最终融合
        fused_3 = fused_13 + fused_23  # 模态 3 的最终融合

        return fused_1, fused_2, fused_3