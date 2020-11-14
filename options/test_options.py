import yaml
import os
import torch
import sys
sys.path.append('../')
from utils import logger

def test_parse(opt):
    path, dataset_name, scale, name = opt.opt, opt.dataset_name, opt.scale, opt.name
    dataset_list = dataset_name.split('+')
    with open(path, 'r') as fp:        
        args = yaml.full_load(fp.read())
    lg = logger(name, 'test_log/{}.log'.format(name), is_test=True)
    
    args['dataset_list'] = dataset_list
    args['name'] = name

    # setting for datasets
    args['scale'] = scale
    args['dataset_name'] = dataset_name
    
       
    dataset_opt = args['datasets']['test']
    dataset_opt['scale'] = scale
    if len(dataset_list) > 0:
        dataroot_HR_list = []
        dataroot_LR_list = []
        for dataset_name in dataset_list:
            dataroot_HR_list.append(dataset_opt['dataroot_HR'].replace('dataset_name', dataset_name))
            dataroot_LR_list.append(dataset_opt['dataroot_LR'].replace('dataset_name', dataset_name).replace('N', str(scale)))
        dataset_opt['dataroot_HR'] = dataroot_HR_list
        dataset_opt['dataroot_LR'] = dataroot_LR_list
    else:
        raise NotImplementedError('No dataset name!')

    # setting for networks
    args['networks']['upscale_factor'] = scale
    args['networks']['which_model'] = opt.which_model
    args['networks']['pretrained'] = opt.pretrained
      
    # setting for GPU environment
    if opt.gpu_ids is not None:
        gpu_list = opt.gpu_ids
    else:
        gpu_list = ','.join([str(x) for x in args['gpu_ids']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list    

    return dict_to_nonedict(args), lg


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        for k, v in opt.items():
            opt[k] = dict_to_nonedict(v)
        return NoneDict(**opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(x) for x in opt]
    else:
        return opt
