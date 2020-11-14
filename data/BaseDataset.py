import torch
import torch.utils.data as data
import random
import numpy as np
import torch

class Base(data.Dataset):
    def __init__(self, opt):
        super(Base, self).__init__()
        
    def __len__(self):
        pass

    def __getitem__(self):
        pass

    # numpy HWC[0:255] -> tensor CHW[0:1].
    def np2tensor(self, img):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255.

        return img

    # get random patch of numpy data
    def get_patch(self, lr, hr, ps, scale):
        lr_h, lr_w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]
        
        lr_x = random.randint(0, lr_w - ps)
        lr_y = random.randint(0, lr_h - ps)
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_patch = lr[lr_y:lr_y+ps, lr_x:lr_x+ps, :]
        hr_patch = hr[hr_y:hr_y+ps*scale, hr_x:hr_x+ps*scale, :]

        return lr_patch, hr_patch
    
    def augment(self, lr, hr, flip=True, rot=True):
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            lr = np.ascontiguousarray(lr[:, ::-1, :])
            hr = np.ascontiguousarray(hr[:, ::-1, :])
        if vflip:
            lr = np.ascontiguousarray(lr[::-1, :, :])
            hr = np.ascontiguousarray(hr[::-1, :, :])
        if rot90:
            lr = lr.transpose(1, 0, 2)
            hr = hr.transpose(1, 0, 2)

        return lr, hr
