import torch
import torch.nn as nn
from .blocks import MeanShift, UpSampler, ResBlock
from torchsummaryX import summary

class EDSR(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=3, num_fea=64, out_channels=3, n_resblocks=16):
        super(EDSR, self).__init__()
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # feature extraction
        self.fea_conv = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # body
        body = []
        for i in range(n_resblocks):
            body.append(ResBlock(num_fea))
        body.append(nn.Conv2d(num_fea, num_fea, 3, 1, 1))
        self.body = nn.Sequential(*body)

        # reconstruction
        self.upsample = nn.Sequential(
            UpSampler(upscale_factor, num_fea),
            nn.Conv2d(num_fea, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        # extract feature
        x = self.fea_conv(x)

        # body
        res = self.body(x)
        res += x

        x = self.upsample(res)
        x = self.add_mean(x)

        return x

'''
# 1518K, 114.2G
s = 4
model = EDSR(upscale_factor=s).to('cuda')
in_ = torch.randn(1, 3, round(720/s), round(1280/s)).to('cuda')
summary(model, in_)
'''
