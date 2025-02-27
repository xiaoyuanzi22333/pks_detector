import torch.nn as nn
import torch
from ..decoder_old import Simple_MLP


class AtNet_decoder(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(AtNet_decoder, self).__init__()
        self.mlp1 = Simple_MLP(in_ch, hid_ch)
        self.mlp2 = Simple_MLP(hid_ch, out_ch)
        
    
    
    def forward(self, fused):
        output = self.mlp1(fused)
        output = self.mlp2(output)
        
        return output