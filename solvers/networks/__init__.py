from .IMDN import IMDN
from .EDSR import EDSR
from .LatticeNet import LatticeNet
from .CARN import CARN
from .IDN import IDN
from .HFAN import HFAN

# ablation studies
from .HFAN_ablation_base import HFAN_ablation_base
from .HFAN_ablation_base_head_tail_conv import HFAN_ablation_base_head_tail_conv
from .HFAN_ablation_base_transformation import HFAN_ablation_base_transformation

# compare with current attention mechanism
from .Baseline import Baseline
from .BaselineCA import BaselineCA
from .BaselineCCA import BaselineCCA
from .BaselineSA import BaselineSA
from .BaselineESA import BaselineESA

def create_model(opt):
    which_model = opt['which_model']
    scale = opt['upscale_factor']
    in_channels = opt['in_channels']
    num_fea = opt['num_fea']
    out_channels = opt['out_channels']

    # State-of-the-art
    if which_model == 'IMDN':
        model = IMDN(scale, in_channels, num_fea, out_channels)
    elif which_model == 'EDSR':
        model = EDSR(scale, in_channels, num_fea, out_channels, opt['n_resblocks'])
    elif which_model == 'LatticeNet':
        model = LatticeNet(scale, in_channels, num_fea, out_channels, opt['num_LBs'])
    elif which_model == 'CARN':
        model = CARN(scale, in_channels, num_fea, out_channels)
    elif which_model == 'IDN':
        model = IDN(scale, in_channels, num_fea, out_channels)
    elif which_model == 'HFAN':
        model = HFAN(scale, in_channels, num_fea, out_channels, opt['down_RBs'], opt['up_RBs'], opt['reduce_rate'])

    # Ablation Study
    elif which_model == 'HFAN_ablation_base':
        model = HFAN_ablation_base(scale, in_channels, num_fea, out_channels, opt['down_RBs'], opt['up_RBs'], opt['reduce_rate'])
    elif which_model == 'HFAN_ablation_base_head_tail_conv':
        model = HFAN_ablation_base_head_tail_conv(scale, in_channels, num_fea, out_channels, opt['down_RBs'], opt['up_RBs'], opt['reduce_rate'])
    elif which_model == 'HFAN_ablation_base_transformation':
        model = HFAN_ablation_base_transformation(scale, in_channels, num_fea, out_channels, opt['down_RBs'], opt['up_RBs'], opt['reduce_rate'])

    # Compare with current attention mechanism
    elif which_model == 'Baseline':
        model = Baseline(scale, in_channels, num_fea, out_channels, opt['down_RBs'])
    elif which_model == 'BaselineCA':
        model = BaselineCA(scale, in_channels, num_fea, out_channels, opt['down_RBs'])
    elif which_model == 'BaselineCCA':
        model = BaselineCCA(scale, in_channels, num_fea, out_channels, opt['down_RBs'])
    elif which_model == 'BaselineSA':
        model = BaselineSA(scale, in_channels, num_fea, out_channels, opt['down_RBs'])
    elif which_model == 'BaselineESA':
        model = BaselineESA(scale, in_channels, num_fea, out_channels, opt['down_RBs'])

    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))

    return model
