import torch
import torch.nn as nn
from .blocks import MeanShift
from torchsummaryX import summary


class ResBlock(nn.Module):
    def __init__(self, num_fea, att=None):
        super(ResBlock, self).__init__()
        self.att = att
        self.conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )
    
    def forward(self, x):
        res = self.conv(x)
        if self.att is not None:
            res = self.att(res)
        
        return res + x


class Baseline(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=47, out_channels=3, down_RBs=15):
        super(Baseline, self).__init__()

        self.down_RBs = down_RBs
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_conv = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # Down RBs
        RBs = []
        for i in range(down_RBs):
            RBs.append(ResBlock(num_fea=num_fea))
        self.RBs = nn.ModuleList(RBs)
    
        # reconstruct
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_fea, out_channels * (upscale_factor**2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        # extract features
        fea = self.fea_conv(x)

        h = fea
        # high frequency learning stage
        for i in range(self.down_RBs):
            h = self.RBs[i](h)

        # reconstruct
        out = self.upsampler(fea + h)
        out = self.add_mean(out)

        return out

'''
# 620K, 35.6G
s = 4
m = Baseline(upscale_factor=s).to('cuda')
i = torch.randn(1, 3, round(720/s), round(1280/s)).to('cuda')
summary(m, i)
'''
