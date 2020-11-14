from .BaseDataset import Base
import numpy as np
import os.path as osp
import pickle

'''
Train or Validate on DIV2KDataset
'''

class DIV2KDataset(Base):
    def __init__(self, opt):
        super(DIV2KDataset, self).__init__(opt)
        self.dataroot_hr = opt['dataroot_HR']
        self.dataroot_lr = opt['dataroot_LR']
        self.filename_path = opt['filename_path']
        self.use_flip = opt['use_flip']
        self.use_rot = opt['use_rot']
        self.ps = opt['patch_size']
        self.scale = opt['scale']
        self.split = opt['split']
        self.enlarge_times = opt['enlarge_times']

        self.img_list = []        
        with open(self.filename_path, 'r') as f:
            filenames = f.readlines()
        for line in filenames:
            self.img_list.append(line.strip())

        
    def __len__(self):
        return len(self.img_list) * self.enlarge_times


    def __getitem__(self, idx):
        idx = idx % len(self.img_list)
        hr_path = osp.join(self.dataroot_hr, self.img_list[idx])
        base, ext = osp.splitext(self.img_list[idx])
        lr_basename = base + 'x{}'.format(self.scale) + '.pt'
        lr_path = osp.join(self.dataroot_lr, lr_basename)
        with open(hr_path ,'rb') as f:
            hr = pickle.load(f)
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)

        data = {}
        if self.split == 'train':
            lr_patch, hr_patch = self.get_patch(lr, hr, self.ps, self.scale)
            lr, hr = self.augment(lr_patch, hr_patch, self.use_flip, self.use_rot)
        lr ,hr = self.np2tensor(lr), self.np2tensor(hr)
        
        data['LR'] = lr
        data['HR'] = hr

        return data
