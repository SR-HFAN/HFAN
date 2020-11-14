import torch
import torch.nn as nn
from .blocks import MeanShift, IMDM
from torchsummaryX import summary

class IMDN(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=3, num_fea=64, out_channels=3):
        super(IMDN, self).__init__()
        
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_conv = nn.Conv2d(in_channels, num_fea, 3, 1, 1)
    
        # map
        self.IMDM1 = IMDM(num_fea)
        self.IMDM2 = IMDM(num_fea)
        self.IMDM3 = IMDM(num_fea)
        self.IMDM4 = IMDM(num_fea)
        self.IMDM5 = IMDM(num_fea)
        self.IMDM6 = IMDM(num_fea)

        self.fuse = nn.Sequential(
            nn.Conv2d(num_fea * 6, num_fea, 1, 1, 0),
            nn.LeakyReLU(0.05)
        )

        self.LR_conv = nn.Conv2d(num_fea, num_fea, 3, 1, 1)

        # reconstruct
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_fea, out_channels * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )
   
    def forward(self, x):
        x = self.sub_mean(x)

        # extract features
        x = self.fea_conv(x)

        # IMDM
        out1 = self.IMDM1(x)
        out2 = self.IMDM2(out1)
        out3 = self.IMDM3(out2)
        out4 = self.IMDM4(out3)
        out5 = self.IMDM5(out4)
        out6 = self.IMDM6(out5)

        out = self.LR_conv(self.fuse(torch.cat([out1, out2, out3, out4, out5, out6], dim=1))) + x

        # reconstruct
        out = self.upsampler(out)

        out = self.add_mean(out)

        return out

'''
# 715K, 40.9G
s = 4
m = IMDN(upscale_factor=s).to('cuda')
in_ = torch.randn(1, 3, round(720/s), round(1280/s)).to('cuda')
summary(m, in_)
'''
