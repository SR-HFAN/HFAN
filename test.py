import argparse
import copy
import torch
from options import test_parse
from utils import logger, calc_cuda_time, calc_metrics
from data import create_dataset, create_loader
from solvers.networks import create_model
from solvers import create_solver
import os
import os.path as osp
import imageio
from torchsummary import summary


def main():
    
    # setting arguments
    parser = argparse.ArgumentParser(description='Test arguments')
    parser.add_argument('--opt', type=str, required=True, help='path to test yaml file')
    parser.add_argument('--name', type=str, required=True, help='test log file name')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--gpu_ids', type=str, default=None, help='which gpu to use')
    parser.add_argument('--which_model', type=str, required=True, help='which pretrained model')
    parser.add_argument('--pretrained', type=str, required=True, help='pretrained path')

    args = parser.parse_args()
    args, lg = test_parse(args)
    pn = 50
    half_pn = pn//2
    lg.info('\n' + '-'*pn + 'General INFO' + '-'*pn)

    # create test dataloader
    test_loader_list = []
    for i in range(len(args['dataset_list'])):
        # get single dataset and dataloader
        single_dataset_args = copy.deepcopy(args)
        single_dataset_args['datasets']['test']['dataroot_HR'] = single_dataset_args['datasets']['test']['dataroot_HR'][i]
        single_dataset_args['datasets']['test']['dataroot_LR'] = single_dataset_args['datasets']['test']['dataroot_LR'][i]
   
        test_dataset = create_dataset(single_dataset_args['datasets']['test'])
        test_loader = create_loader(test_dataset, args['datasets']['test'])
        test_loader_list.append(test_loader)

    # create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(args['networks']).to(device)
    lg.info('Create model: [{}]'.format(args['networks']['which_model']))
    scale = args['scale']

    # calc number of parameters
    in_ = torch.randn(1, 3, round(720/scale), round(1280/scale)).to(device)
    params, GFlops = summary(model, in_)
    lg.info('Total parameters: [{:.3f}M], GFlops: [{:.4f}G]'.format(params / 1e6, GFlops / 1e9))

    # load pretrained model
    state_dict = torch.load(args['networks']['pretrained'])
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    lg.info('Load pretrained from: [{}]'.format(args['networks']['pretrained']))

    for i, test_loader in enumerate(test_loader_list):
        dataset_name = args['dataset_list'][i]
        l = (12-len(dataset_name)) // 2
        e = len(dataset_name) % 2
        lg.info('\n\n' + '-'*(pn+l) + dataset_name + '-'*(pn+l+e))
        lg.info('Number of [{}] images: [{}]'.format(dataset_name, len(test_loader)))

        # calculate cuda time
        avg_test_time = 0.0
        if args['calc_cuda_time'] and 'Set5' in dataset_name:
            lg.info('Start calculating cuda time...')
            avg_test_time = calc_cuda_time(test_loader, model)
            lg.info('Average cuda time on [{}]: [{:.5f}ms]'.format(dataset_name, avg_test_time))
    
        # calucate psnr and ssim
        psnr_list = []
        ssim_list = []

        model.eval()
        for iter, data in enumerate(test_loader):
            lr = data['LR'].to(device)
            hr = data['HR']
        
            # calculate evaluation metrics
            with torch.no_grad():
                sr = model(lr)
            #save(args['networks']['which_model'], dataset_name, data['filename'][0], tensor2np(sr))
            psnr, ssim = calc_metrics(tensor2np(sr), tensor2np(hr), crop_border=scale, test_Y=True)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            lg.info('[{:03d}/{:03d}] || PSNR/SSIM: {:.2f}/{:.4f} || {}'.format(iter+1, len(test_loader), psnr, ssim, data['filename']))


        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)

        if avg_test_time > 0:
            lg.info('Average PSNR: {:.2f}  Average SSIM: {:.4f}, Average time: {:.5f}ms'.format(avg_psnr, avg_ssim, avg_test_time))
        else:
            lg.info('Average PSNR: {:.2f}  Average SSIM: {:.4f}'.format(avg_psnr, avg_ssim))
        
    lg.info('\n' + '-'*pn + '---Finish---' + '-'*pn)


def tensor2np(t): # CHW -> HWC, [0, 1] -> [0, 255]
    return (t[0]*255).cpu().clamp(0, 255).round().byte().permute(1, 2, 0).numpy()


def save(which_model, dataset, filename ,sr):
    filepath = osp.join('results', which_model, dataset, filename)
    if not osp.exists(osp.dirname(filepath)):
        os.makedirs(osp.dirname(filepath)) 
    imageio.imwrite(filepath, sr)    


if __name__ == '__main__':
    main()
