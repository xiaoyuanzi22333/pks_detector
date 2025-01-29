import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import split_tensor


# basic structure of Resnet block
class ResBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):

        super(ResBasicBlock, self).__init__()

        self.cbl1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
        )

        self.cbl2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

        self.lrelu = nn.LeakyReLU(0.1)
    

    def forward(self, input):
        identity = input
        print(input.shape)
        output1 = self.cbl1(input)
        output2 = self.cbl2(output1)
        output = output2 + identity
        output = self.lrelu(output)

        return output



# encoder to process splited data
class Res_encoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, window_size=15, step=5):
        super(Res_encoder, self).__init__()
        self.window_size = 15
        self.step = 5

        self.ResBlock1 = ResBasicBlock(in_ch, out_ch)
        self.ResBlock2 = ResBasicBlock(out_ch, out_ch)
        self.ResBlock3 = ResBasicBlock(out_ch, out_ch)

        self.lstm = nn.LSTM(window_size*out_ch,window_size*out_ch*2,num_layers=2)
        

    def forward(self, input):
        # input [L,1]
        output = self.ResBlock1(input)
        output = self.ResBlock2(output)
        output = self.ResBlock3(output)
        # output [L,3]
        splited_output = split_tensor(output, self.window_size, self.step)
        print("splited_output: " + str(splited_output.shape))
        # splited_output [N,15*3]
        output, (hn, cn) = self.lstm(splited_output)
        # output [N, 15*3*2]
        # return the last output of LSTM  [15*3*2]
        return output[-1]

