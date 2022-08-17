import torch
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.core import psnr, ssim, tensor2img
from mmcv import Config

def load_mmedit_dataset(cfg_path: str, training: bool):
    # cfg = Config.fromfile('configs/editing/DIV2K.py')
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
        samples_per_gpu=cfg.data.test_dataloader.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        pin_memory=False)
    eval_kwargs = cfg.get('evaluation', {}).copy()
    return dataset, data_loader, {}

def mmedit_process_output(outputs, gt):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}
    test_cfg = {'metrics': ['PSNR', 'SSIM'], 'crop_border':4}
    crop_border = test_cfg['crop_border']
    output = tensor2img(torch.tensor(outputs['output']))
    gt = tensor2img(gt)
    eval_result = dict()
    for metric in test_cfg['metrics']:
        eval_result[metric] = allowed_metrics[metric](output, gt, crop_border)
    return eval_result

def mmedit_collate_fn(data, device: str):
    return data['lq'].to(device)