from . import decoder
from . import encoder
import torch
import torch.nn as nn


class pksNet(nn.modules):
    def __init__(self, in_ch=1,hid_ch=3, out_ch=3, window_size=15, step=5):
        super(pksNet, self).__init__()

        self.brake_encoder = encoder.Res_encoder(in_ch, hid_ch)
        self.steer_encoder = encoder.Res_encoder(in_ch, hid_ch)
        self.throttle_encoder = encoder.Res_encoder(in_ch, hid_ch)

        self.brake_decoder = decoder.simple_Decoder(window_size*hid_ch*2 ,window_size*hid_ch, out_ch)
        self.steer_decoder = decoder.simple_Decoder(window_size*hid_ch*2, window_size*hid_ch, out_ch)
        self.throttle_decoder = decoder.simple_Decoder(window_size*hid_ch*2, window_size*hid_ch, out_ch)

        self.mlp = decoder.simple_MLP(out_ch, 2)

    

    def forward(self, brake, steer, throotle):
        # input [batchsize, L, 1]
        encode1 = self.brake_encoder(brake)
        encode2 = self.steer_encoder(steer)
        encode3 = self.throttle_encoder(throotle)
        # encode [batchsize, windowsize*hidch*2]
        output1 = self.brake_decoder(encode1)
        output2 = self.steer_decoder(encode2)
        output3 = self.throttle_decoder(encode3)
        # output [batchsize, outch]
        output = output1 + output2 + output3
        # output [batchsize, outch]
        output = self.mlp(output)
        # output [batchsize, 2]
        return output



