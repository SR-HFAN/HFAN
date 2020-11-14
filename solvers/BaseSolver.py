import torch
import torch.nn as nn
from torch.nn import init
import functools
from torchsummary import summary
import math

class BaseSolver(object):
    def __init__(self, opt):
        self.opt = opt
        self.ps = opt['patch_size']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # skip unstable batch for stable training
        self.last_epoch_loss = 1e8
        self.skip_threshold = opt['solver']['skip_threshold']        
        
        # experiment dirs
        self.exp_root = opt['paths']['experiment_root']
        self.ckp_path = opt['paths']['epochs']
        self.records_path = opt['paths']['records']
        
        # log and visual scheme
        self.save_ckp_step = opt['solver']['save_ckp_step']
        
        self.best_epoch = 0
        self.cur_epoch = 0
        self.best_pred = 0.0

    def count_parameters_GFlops(self):
        height = round(720 / self.scale)
        width = round(1280 / self.scale)

        # count GFlops
        in_ = torch.randn(1, 3, height, width).to(self.device)
        params, GFlops = summary(self.model, in_)

        return params, GFlops


    def init_weight(self, model, init_type='kaiming', scale=1, std=0.02):
        if init_type == 'kaiming':
            init_func = functools.partial(self.init_weight_kaiming, scale=scale)
        elif init_type == 'normal':
            init_func = functools.partial(self.init_weight_normal, std=std)
        elif init_type == 'uniform':
            init_func = functools.partial(self.init_weight_uniform)
        elif init_type == 'orthogonal':
            init_func = self.init_weight_orthogonal
        else:
            raise NotImplementedError('Initialization method [{}] is not implemented!'.format(init_type))
        model.apply(init_func)


    def init_weight_kaiming(self, m, scale=1):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
           if classname != 'MeanShift':
                init.kaiming_normal_(m.weight.data) 
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.constant_(m.weight.data, 1.0)
            m.weight.data *= scale
            init.constant_(m.bias.data, 0.0)

    
    def init_weight_normal(self, m, std=0.02):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != 'MeanShift':
                init.normal_(m.weight.data, 0.0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, std)
            init.constant_(m.bias.data, 0.0)


    def init_weight_uniform(self, m):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != 'MeanShift':
                in_channels = m.weight.data.shape[1]
                k = 1.0 / math.sqrt(in_channels)
                init.uniform_(m.weight.data, -k, k)
                if m.bias is not None:
                    init.uniform_(m.bias.data, -k, k)
        elif isinstance(m, (nn.Linear)):
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, std)
            init.constant_(m.bias.data, 0.0)


    def init_weight_orthogonal(self, m):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != 'MeanShift':
                init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
