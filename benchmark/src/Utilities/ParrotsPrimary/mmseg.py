import numpy as np
import torch
from mmcv import Config

from mmseg.datasets import build_dataloader, build_dataset

def load_mmseg_dataset(cfg_path: str, training: bool):
    # 'configs/segmentation/cityscapes.py'
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=2,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        pin_memory=False,
        shuffle=False)
    eval_kwargs = cfg.get('evaluation', {}).copy()
    return dataset, data_loader, eval_kwargs

def mmseg_process_output(outputs, batch_size, img_metas, img_shape):
    h,w,_ = img_metas[0]["pad_shape"]
    output = outputs['output'].reshape((batch_size,1,h,w))
    batch_results = []
    for i in range(batch_size):
        batch_results.append(output[i][0])
    return batch_results
