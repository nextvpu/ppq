
import numpy as np
import torch
from mmcv import Config

from mmdet.datasets import build_dataloader, build_dataset

def load_mmdet_dataset(cfg_path: str, training: bool):
    cfg = Config.fromfile(cfg_path)
    print("cfg.data.test: ", cfg.data.test)
    dataset = build_dataset(cfg.data.test) if not training else build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    eval_kwargs = {}
    return dataset, data_loader, eval_kwargs

def mmdet_process_output(outputs, batch_size, img_metas, img_shape):
    from mmdet.core import bbox2result, encode_mask_results
    
    batch_dets,batch_labels = outputs['dets'],outputs['labels']
    batch_masks = outputs.get('masks',None)
    batch_dets = batch_dets.reshape((batch_size, -1, 5))
    batch_labels = batch_labels.reshape((batch_size, -1))
    batch_results = []
    img_h, img_w = img_shape[2:]

    if batch_masks is not None:
        batch_masks = batch_masks.reshape((batch_size, -1, img_h, img_w))
    for j in range(batch_size):
        dets, labels = batch_dets[j], batch_labels[j]
        scale_factor = img_metas[j]['scale_factor']
        dets[:, :4] /= scale_factor
        if 'border' in img_metas[j]:
            x_off = img_metas[j]['border'][2]
            y_off = img_metas[j]['border'][0]
            dets[:, [0, 2]] -= x_off
            dets[:, [1, 3]] -= y_off
            dets[:, :4] *= (dets[:, :4] > 0).astype(dets.dtype)
        dets_results = bbox2result(dets, labels, 80)
        if batch_masks is not None:
            masks = batch_masks[j]
            img_h, img_w = img_metas[j]['img_shape'][:2]
            ori_h, ori_w = img_metas[j]['ori_shape'][:2]
            masks = masks[:, :img_h, :img_w]
            masks = masks.astype(np.float32)
            masks = torch.from_numpy(masks)
            masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(ori_h, ori_w))
            masks = masks.squeeze(0).detach().numpy()
            if masks.dtype != np.bool:
                masks = masks >= 0.5
            segms_results = [[] for _ in range(80)]
            for k in range(len(dets)):
                segms_results[labels[k]].append(masks[k])
            batch_results.append((dets_results, segms_results))
        else:
            batch_results.append(dets_results)
    if isinstance(batch_results[0], tuple):
        batch_results = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in batch_results]
    return batch_results

def collate_fn(data, device: str):
    try:
        img = data['img']
        assert isinstance(img, torch.Tensor)
        if img.ndim == 3:
            return img.to(device).unsqueeze(0)
        if img.ndim == 4:
            return img.to(device)
    except Exception as _: pass

    try:
        img = data['img'][0]
        assert isinstance(img, torch.Tensor)
        if img.ndim == 3:
            return img.to(device).unsqueeze(0)
        if img.ndim == 4:
            return img.to(device)
    except Exception as _: pass

    try:
        img = data['img'][0].data[0]
        assert isinstance(img, torch.Tensor)
        if img.ndim == 3:
            return img.to(device).unsqueeze(0)
        if img.ndim == 4:
            return img.to(device)
    except Exception as _: pass

def meta_collate_fn(data):
    if isinstance(data['img_metas'], list):
        img_metas = data['img_metas'][0].data[0]
    else:
        img_metas = data['img_metas'].data[0]
    return img_metas
