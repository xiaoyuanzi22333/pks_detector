import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np



class Simple_MLP(nn.Module):
    def __init__(self, in_ch, out_ch): 
        super(Simple_MLP, self).__init__()

        self.ln = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.ru = nn.LeakyReLU(0.1)

    
    def forward(self, input):
        output = self.ln(input)
        output = self.bn(output)
        output = self.ru(output)

        return output
    


class Simple_Decoder(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(Simple_Decoder, self).__init__()

        self.mlp1 = Simple_MLP(in_ch, hid_ch)
        self.mlp2 = Simple_MLP(hid_ch, out_ch)

    
    def forward(self, input):
        # input [windowsize*encoderoutch*2]
        output = self.mlp1(input)
        # output [hid_ch]
        output = self.mlp2(output)
        # output [out_ch]

        return output