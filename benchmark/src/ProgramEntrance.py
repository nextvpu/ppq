# use this script to run all pplnn test cases.
import os
from functools import partial
from typing import Dict, List

import torch
from tqdm import tqdm

from ppq import (QuantableOperation, QuantizationSettingFactory,
                 TargetPlatform, TorchExecutor, convert_any_to_numpy,
                 graphwise_error_analyse)
from ppq.api import export_ppq_graph, quantize_onnx_model
from ppq.core import QuantizationStates
from ppq.utils.fetch import tensor_random_fetch
from Utilities.ParrotsPrimary.mmdet import (collate_fn, load_mmdet_dataset,
                                            meta_collate_fn,
                                            mmdet_process_output)
from Utilities.ParrotsPrimary.mmedit import (load_mmedit_dataset,
                                             mmedit_process_output)
from Utilities.ParrotsPrimary.mmseg import (load_mmseg_dataset,
                                            mmseg_process_output)


CFG_NVP = True                                          # 开关，控制是否执行NVP流程

CFG_MODELS = {

    # DETECTION MODELS:
    'faster_rcnn': {
        'Model': '../models/det_model/faster_rcnn.onnx',
        'Type': 'Detection',
        'InputShape': [1, 3, 1333, 800]
    },
    'fsaf': {
        'Model': '../models/det_model/fsaf.onnx',
        'Type': 'Detection',
        'InputShape': [1, 3, 1333, 800]
    },
    'mask_rcnn': {
        'Model': '../models/det_model/mask_rcnn.onnx',
        'Type': 'Detection',
        'InputShape': [1, 3, 1333, 800]
    },
    'retinanet': {
        'Model': '../models/det_model/retinanet.onnx',
        'Type': 'Detection',
        'InputShape': [1, 3, 1333, 800]
    },
    # SEGMENTATION MODELS:
    'deeplabv3plus': {
        'Model': '../models/seg_model/deeplabv3plus_whole_mode.onnx',
        'Type': 'Segmentation',
        'InputShape': [1, 3, 2048, 1024]
    },
    'deeplabv3': {
        'Model': '../models/seg_model/deeplabv3_whole_mode.onnx',
        'Type': 'Segmentation',
        'InputShape': [1, 3, 2048, 1024]
    },
    'fcn': {
        'Model': '../models/seg_model/fcn_whole_mode.onnx',
        'Type': 'Segmentation',
        'InputShape': [1, 3, 2048, 1024]
    },
    'pspnet': {
        'Model': '../models/seg_model/pspnet_whole_mode.onnx',
        'Type': 'Segmentation',
        'InputShape': [1, 3, 2048, 1024]
    },

    # edit
    'esrgan':{
        'Model': '../models/edit_model/esrgan.onnx',
        'Type': 'Edit',
        'InputShape': [1, 3, 126, 126]
    },
    'srcnn':{
        'Model': '../models/edit_model/srcnn.onnx',
        'Type': 'Edit',
        'InputShape': [1, 3, 126, 126]
    }
}

CFG_DEVICE = 'cuda'
CFG_REQUIRES = {
    'EXPORT_MODEL': True,
    'ERROR_ANALYSE': True,
    'EVALUATION' : True,
    'EXPORT_SAMPLE_VALUE': False
}

CFG_DEVICE = 'cuda'

if CFG_NVP == True:
    TARGET_PLATFORM   = TargetPlatform.NVP_163_INT8
    SETTING = QuantizationSettingFactory.nvp_setting()
else:
    TARGET_PLATFORM = TargetPlatform.PPL_CUDA_INT8
    SETTING = QuantizationSettingFactory.pplcuda_setting()

SETTING.fusion_setting.force_alignment_overlap = True
SETTING.quantize_activation_setting.calib_algorithm = 'kl'

SETTING.dispatcher = 'pplnn'            # Just for Detection
#SETTING.dispatcher = 'conservative'    # Just for Segmentation and Edit

def convert_output_as_dict(output_names: List[str], output: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in zip(output_names, output)}

if __name__ == '__main__':
    for model in CFG_MODELS:
        print(f'Start Quantization Procedure with {model}')
        onnx_path, model_type, shape = (
            CFG_MODELS[model]['Model'], CFG_MODELS[model]['Type'], CFG_MODELS[model]['InputShape'])

        if model_type == 'Detection':
            dataset, test_data_loader, eval_kwargs = load_mmdet_dataset(
                cfg_path='Utilities/ParrotsPrimary/configs/detection.py', training=False)

        elif model_type == 'Segmentation':
            dataset, test_data_loader, eval_kwargs = load_mmseg_dataset(
                cfg_path='Utilities/ParrotsPrimary/configs/segmentation.py', training=False)

        elif model_type == 'Edit':
            dataset, test_data_loader, eval_kwargs = load_mmedit_dataset(
                cfg_path='Utilities/ParrotsPrimary/configs/edit.py', training=False)

        # for mask rcnn detection
        metric = 'bbox'
        # if 'mask' in model:
        #     metric = ['bbox', 'segm']
        #     eval_kwargs.update(dict(metric = metric))
        
        if model_type == 'Edit':
            from Utilities.ParrotsPrimary.mmedit import mmedit_collate_fn
            collate_fn = mmedit_collate_fn

        quant_ir = quantize_onnx_model(
            onnx_import_file=onnx_path, setting=SETTING,
            calib_dataloader=test_data_loader,
            calib_steps=32, input_shape=shape,
            platform=TARGET_PLATFORM,
            collate_fn=partial(collate_fn, device=CFG_DEVICE), 
            do_quantize=True, device=CFG_DEVICE)

        if CFG_REQUIRES['ERROR_ANALYSE']:
            # extract first 16 img from dataloader 
            loader = []
            for data in test_data_loader:
                loader.append(data)
                if len(loader) >= 16: break

            snr_report = graphwise_error_analyse(
                graph=quant_ir, running_device=CFG_DEVICE, dataloader=loader,
                collate_fn=partial(collate_fn, device=CFG_DEVICE))

        if CFG_REQUIRES['EVALUATION']:
            executor = TorchExecutor(graph=quant_ir, device=CFG_DEVICE)
            results_collector = []
            output_processer = {
                'Detection': mmdet_process_output,
                'Segmentation': mmseg_process_output,
                'Edit': mmedit_process_output
            }[model_type]

            for data in tqdm(test_data_loader, total=len(test_data_loader), desc='Evaluating ...'):
                
                if model_type != 'Edit':
                    meta   = meta_collate_fn(data)
                    data   = collate_fn(data, device=CFG_DEVICE)
                    output = executor.forward(data)
                    output = [convert_any_to_numpy(tensor) for tensor in output]
                    output = convert_output_as_dict([name for name in quant_ir.outputs], output)
                    output = output_processer(output, batch_size=data.shape[0], 
                                              img_metas=meta, img_shape=data.shape)
                    results_collector.extend(output)
                else:
                    lq_img, gt_img = data['lq'], data['gt']
                    output = executor.forward(lq_img.to(CFG_DEVICE))
                    output = convert_output_as_dict([name for name in quant_ir.outputs], output)
                    output = output_processer(output, gt_img)
                    results_collector.append(dict(eval_result=output))

            print(dataset.evaluate(results_collector, **eval_kwargs))

        if CFG_REQUIRES['EXPORT_SAMPLE_VALUE']:
            
            def shape_to_str(shape: List[int]) -> str:
                if len(shape) == 1:
                    return str(shape[0])
                string_builder = str(shape[0])
                for s in shape[1: ]:
                    string_builder += '_' + str(s)
                return string_builder
            
            executor = TorchExecutor(graph=quant_ir, device=CFG_DEVICE)
            if not os.path.exists(f'Output/Samples/{model}'):
                os.mkdir(f'Output/Samples/{model}')
            
            if not os.path.exists(f'Output/Samples/{model}/fetchs'):
                os.mkdir(f'Output/Samples/{model}/fetchs')
            
            # fetch one data batch.
            for data in test_data_loader:
                data = collate_fn(data, device=CFG_DEVICE)
                break
            
            interested_vars = set()
            for operation in quant_ir.operations.values():
                if not isinstance(operation, QuantableOperation): continue
                
                for output_var, output_config in zip(
                    operation.outputs, operation.config.output_quantization_config):
                    if output_config.state != QuantizationStates.OVERLAPPED:
                        interested_vars.add(output_var.name)
            interested_vars.update(quant_ir.outputs.keys())
            interested_vars = list(interested_vars)
            
            outputs = executor.forward(inputs=data, output_names=interested_vars)
            for value, name in zip(outputs, interested_vars):

                if name in quant_ir.outputs:
                    np_value = convert_any_to_numpy(value)
                    np_value.tofile(f'Output/Samples/{model}/{name}_{shape_to_str(np_value.shape)}.bin')

                value = tensor_random_fetch(value, num_of_fetches=1024)
                np_value = convert_any_to_numpy(value)
                np_value.tofile(f'Output/Samples/{model}/fetchs/{name}.bin')
            
            assert len(quant_ir.inputs) == 1, 'Oops'
            for key in quant_ir.inputs:
                np_value = convert_any_to_numpy(data)
                np_value.tofile(f'Output/Samples/{model}/{key}_{shape_to_str(np_value.shape)}.bin')

        if CFG_REQUIRES['EXPORT_MODEL']:
            export_ppq_graph(graph=quant_ir, platform=TARGET_PLATFORM, 
                             graph_save_to='Output/{model_name}'.format(model_name=model), 
                             config_save_to='Output/{model_name}.json'.format(model_name=model))