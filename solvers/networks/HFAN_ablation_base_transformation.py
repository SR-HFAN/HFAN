import torch
import torch.nn as nn
from .blocks import MeanShift, ResBlockLReLU
from torchsummaryX import summary

def make_layers(block, num, **kw):
    m = []
    for i in range(num):
        m.append(block(**kw))
    return nn.Sequential(*m)


# High Frequency Attention Block
class HFAB(nn.Module):
    def __init__(self, num_fea=64, up_RBs=2, reduce_rate=2):
        super(HFAB, self).__init__()
        up_fea = num_fea // reduce_rate
        self.transition_in = nn.Conv2d(num_fea, up_fea, 1, 1, 0)
        self.RBs = make_layers(ResBlockLReLU, up_RBs, num_fea=up_fea, mid_fea=up_fea)
        self.transition_out = nn.Conv2d(up_fea, num_fea, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.transition_in(x)
        out = self.RBs(out)
        out = self.transition_out(out)
        att = self.sigmoid(out)
        
        return torch.mul(x, att)


class HFAN_ablation_base_transformation(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=64, out_channels=3, down_RBs=4, up_RBs=2, reduce_rate=2):
        super(HFAN_ablation_base_transformation, self).__init__()

        self.down_RBs = down_RBs
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_conv = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # transition
        self.transition = ResBlockLReLU(num_fea=num_fea, mid_fea=num_fea)
        self.pre_HFAB = HFAB(num_fea, up_RBs, reduce_rate)

        # Down RBs
        RBs = []
        for i in range(down_RBs):
            RBs.append(ResBlockLReLU(num_fea=num_fea, mid_fea=num_fea))
        self.RBs = nn.ModuleList(RBs)
    
        # High Frequency Attention Block
        HFABs = []
        for i in range(down_RBs):
            HFABs.append(HFAB(num_fea, up_RBs, reduce_rate))
        self.HFABs = nn.ModuleList(HFABs)

        # fuse
        self.fuse = nn.Sequential(
            nn.Conv2d(num_fea * down_RBs, num_fea, 1, 1, 0),
            nn.LeakyReLU(0.05)
        )

        # reconstruct
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_fea, out_channels * (upscale_factor**2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        # extract features
        fea = self.fea_conv(x)

        h_list = []

        # transition
        trans_fea = self.transition(fea)
        h = self.pre_HFAB(trans_fea)

        # high frequency learning stage
        for i in range(self.down_RBs):
            out = self.RBs[i](h)
            h = self.HFABs[i](out)
            h_list.append(h)

        # fuse
        concat_h = h_list[0]
        for i in range(self.down_RBs - 1):
            concat_h = torch.cat([concat_h, h_list[i+1]], dim=1)
        h_fuse = self.fuse(concat_h)
        h = h_fuse

        # reconstruct
        out = self.upsampler(fea + h)
        out = self.add_mean(out)

        return out

'''
# 621K, 35.7G
s = 4
m = HFAN_ablation_base_transformation(upscale_factor=s).to('cuda')
i = torch.randn(1, 3, round(720/s), round(1280/s)).to('cuda')
summary(m, i)
'''
