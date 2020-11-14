import yaml
import os
import os.path as osp
import logging
import sys
sys.path.append('../')
from utils import logger
import shutil

def parse(opt):
    path, name, resume = opt.opt, opt.name, opt.resume
    
    with open(path, 'r') as fp:        
        args = yaml.full_load(fp.read())
    lg = logger(name, 'log/{}.log'.format(name), resume)
        
    # general settings
    args['name'] = name
    args['scale'] = opt.scale
    args['use_chop'] = opt.use_chop

    # setting for datasets
    scale = opt.scale
       
    for phase, dataset_opt in args['datasets'].items():
        dataset_opt['scale'] = scale
        dataset_opt['split'] = phase
        dataset_opt['patch_size'] = opt.ps
        dataset_opt['batch_size'] = opt.bs
        if 'DIV2K' in dataset_opt['dataroot_LR']:
            dataset_opt['dataroot_LR'] = dataset_opt['dataroot_LR'].replace('N', str(opt.scale))        

    # setting for networks
    args['networks']['upscale_factor'] = scale

    # setting for solver
    args['solver']['learning_rate'] = opt.lr
    
      
    # setting for GPU environment
    if opt.gpu_ids is not None:
        gpu_list = opt.gpu_ids
    else:
        gpu_list = ','.join([str(x) for x in args['gpu_ids']])
    lg.info('Available gpus: {}'.format(gpu_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list    
    if ',' in gpu_list:
        args['networks']['dataparallel'] = True
        lg.info('Using Dataparallel')

    # create directories for paths
    args['solver']['resume'] = resume

    root = osp.join(args['paths']['experiment_root'], name)
    args['paths']['root'] = root
    args['paths']['epochs'] = osp.join(root, 'epochs')
    args['paths']['records'] = osp.join(root, 'records')
    
    if osp.exists(root) and resume is None:
        lg.info('Remove dir: [{}]'.format(root))
        shutil.rmtree(root, True)    
    
    for name, path in args['paths'].items():
        if not osp.exists(path):
            os.mkdir(path)
            lg.info('Create directory: {}'.format(path))
 
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
