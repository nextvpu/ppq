#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import os
import os.path as osp
import random
import json
import time

from multiprocessing.pool import Pool

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

# Parameters
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
# Get orientation exif tag
for k,v in ExifTags.TAGS.items():
    if v == 'Orientation':
        ORIENTATION = k
        break


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


class TrainValDataset(Dataset):
    # YOLOv6 train_loader/val_loader, loads images and labels for training and validation
    def __init__(self, img_dir, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, 
                 check_images=False, check_labels=False, stride=32, pad=0.0, rank=-1, class_names=None, 
                 task='train', cache_path: str=None):
        assert task.lower() in ('train', 'val', 'speed'), f'Not supported task: {task}'
        t1 = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1,0)
        self.task = self.task.capitalize()
        self.cache_path = cache_path
        
        self.img_paths = []
        for file in os.listdir(img_dir):
            self.img_paths.append(os.path.join(img_dir, file))

        if self.rect:
            shapes = [self.img_info[p]['shape'] for p in self.img_paths]
            self.shapes = np.array(shapes, dtype=np.float64)
            self.batch_indices = np.floor(np.arange(len(shapes)) / self.batch_size).astype(np.int)   # batch indices of each image
            self.sort_files_shapes()
        t2 = time.time()
        if self.main_process:
            print(f'%.1fs for dataset initialization.'%(t2-t1))

    def __len__(self):
        '''Get the length of dataset'''
        return len(self.img_paths)

    def __getitem__(self, index):
        '''Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        '''
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)

        # Letterbox
        shape = self.batch_shapes[self.batch_indices[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img)

    def load_image(self, index):
        '''Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.
        Returns:
            Image, original shape of image, resized image shape
        '''
        path = self.img_paths[index]
        im = cv2.imread(path)
        assert im is not None, f'Image Not Found {path}, workdir: {os.getcwd()}'

        h0, w0 = im.shape[:2]  # origin shape
        r = self.img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]

    @staticmethod
    def collate_fn(batch):
        '''Merges a list of samples to form a mini-batch of Tensor(s)'''
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def sort_files_shapes(self):
        # Sort by aspect ratio
        batch_num = self.batch_indices[-1]+1
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_paths = [self.img_paths[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * batch_num
        for i in range(batch_num):
            ari = ar[self.batch_indices == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(np.int) * self.stride

    @staticmethod
    def check_image(im_file):
        # verify an image.
        nc, msg = 0, ''
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = im.size  # (width, height)
            im_exif = im._getexif()
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6,8):
                    shape = (shape[1], shape[0])

            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            if im.format.lower() in ('jpg', 'jpeg'):
                with open(im_file, 'rb') as f:
                    f.seek(-2, 2)
                    if f.read() != b'\xff\xd9':  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                        msg += f'WARNING: {im_file}: corrupt JPEG restored and saved'
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f'WARNING: {im_file}: ignoring corrupt image: {e}'
            return im_file, None, nc, msg

    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc,  msg = 0, 0, 0, 0,''  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, 'r') as f:
                    labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(len(l) == 5 for l in labels), f'{lb_path}: wrong label format.'
                    assert (labels >= 0).all(), f'{lb_path}: Label values error: all values in label file must > 0'
                    assert (labels[:, 1:] <= 1).all(), f'{lb_path}: Label values error: all coordinates must be normalized'

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f'WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed'
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path,labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f'WARNING: {lb_path}: ignoring invalid labels: {e}'
            return None, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(img_info,class_names,save_path):
        # for evaluation with pycocotools
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, class_name in enumerate(class_names):
            dataset['categories'].append({'id': i, 'name': class_name, 'supercategory': ''})

        ann_id = 0
        print(f'Convert to COCO format')
        for i, (img_path,info) in enumerate(tqdm(img_info.items())):
            labels = info['labels'] if info['labels'] else []
            img_id = osp.basename(img_path).split('.')[0]
            img_h,img_w = info['shape']
            dataset['images'].append({'file_name': os.path.basename(img_path),
                                        'id': img_id,
                                        'width': img_w,
                                        'height': img_h})
            if labels:
                for label in labels:
                    c,x,y,w,h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset['annotations'].append({
                        'area': h*w,
                        'bbox': [x1, y1, w, h],
                        'category_id': cls_id,
                        'id': ann_id,
                        'image_id': img_id,
                        'iscrowd': 0,
                        # mask
                        'segmentation': []
                    })
                    ann_id += 1

        with open(save_path, 'w') as f:
            json.dump(dataset, f)
            print(f'Convert to COCO format finished. Resutls saved in {save_path}')