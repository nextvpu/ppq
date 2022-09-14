import cv2
import argparse
import onnx
import numpy as np
import onnxruntime
import torch
from tqdm import tqdm
import random

from ppq import *
from ppq.api import *

from NVPINT8Quantizer import *
from NVPINT8Quantizer import nvp_quant_setting

QUANT_PLATFROM = TargetPlatform.EXTENSION

def get_calib_dataloader(args):
    shape = args.shape
    calibration = args.calibration
    mean = args.mean
    std = args.std
    colorformat = args.colorformat

    shape_map = dict()
    for i in shape.split(" "):
        if not i :
            continue
        tnsr_name, _shape = i.split(':')
        shape_map[tnsr_name] = [int(_) for _ in _shape.split('x')]

    mean_map = dict()
    for i in mean.split(" "):
        if not i :
            continue
        tnsr_name, _mean = i.split(':')
        mean_map[tnsr_name] = [float(_) for _ in _mean.split(',')]
        assert len(mean_map[tnsr_name]) == shape_map[tnsr_name][1], 'dimension mismatch.'

    std_map = dict()
    for i in std.split(" "):
        if not i :
            continue
        tnsr_name, _std = i.split(':')
        std_map[tnsr_name] = [float(_) for _ in _std.split(',')]
        assert len(std_map[tnsr_name]) == shape_map[tnsr_name][1], 'dimension mismatch.'

    path_map = dict()
    with open(calibration, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            tnsr_name, _path = line.split('=')
            _path = _path.replace('\n', '')
            if tnsr_name not in path_map:
                path_map[tnsr_name] = [_path]
            else:
                path_map[tnsr_name].append(_path)
    
    assert len(path_map) == 1, 'Currently only single input onnx model is supported for quantization.'
    tnsr_name  = list(shape_map.keys())[0]
    tnsr_shape = shape_map[tnsr_name]
    channel = tnsr_shape[1]
    img_mean = np.array(mean_map[tnsr_name]).reshape([1, channel, 1, 1])
    img_std = np.array(std_map[tnsr_name]).reshape([1, channel, 1, 1])
    
    if channel == 1:
        assert colorformat == 'Gray', 'The channel is 1, so the format should be Gray.'
        img_data = [cv2.resize(cv2.imdecode(np.fromfile(_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE), (tnsr_shape[3], tnsr_shape[2])).reshape(tnsr_shape).astype(np.float32) for _path in path_map[tnsr_name]]
    else:
        if colorformat == 'RGB':
            img_data = [cv2.resize(cv2.imdecode(np.fromfile(_path, dtype=np.uint8), cv2.IMREAD_COLOR), (tnsr_shape[3], tnsr_shape[2]))[:,:,[2,1,0]].transpose((2,0,1)).reshape(tnsr_shape).astype(np.float32) for _path in path_map[tnsr_name]]
        elif colorformat == 'BGR':
            img_data = [cv2.resize(cv2.imdecode(np.fromfile(_path, dtype=np.uint8), cv2.IMREAD_COLOR), (tnsr_shape[3], tnsr_shape[2])).transpose((2,0,1)).reshape(tnsr_shape).astype(np.float32) for _path in path_map[tnsr_name]]
        else:
            assert False, 'Invaild color format: {}'.format(colorformat)
    img_data = [((_img-img_mean)/img_std).astype(np.float32) for _img in img_data]

    calib_dataloader = [torch.from_numpy(npy_tensor) for npy_tensor in img_data]

    return tnsr_shape, calib_dataloader

# cv2.imdecode(np.fromfile(_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def main(args):
    input = args.input
    outdir = args.outdir
    device = args.device
    layerwise_analyse = args.layerwise_analyse
    finetune = args.finetune
    no_quant_network_output = args.no_quant_network_output
    no_quant_op_types = args.no_quant_op_types 

    assert os.path.exists(input), 'onnx model "{}" file not found.'.format(input)

    # 1.initialize dataloader
    input_shape, calib_dataloader = get_calib_dataloader(args)

    # 2.initialize quantization setting
    quant_setting = nvp_quant_setting
    onnx_model = onnx.load(input)
    wgt_dims = {}
    for w in onnx_model.graph.initializer:
        wgt_dims[w.name] = list(w.dims)

    for node in onnx_model.graph.node:
        if node.op_type == "Conv":
            group = [_ for _ in node.attribute if _.name == 'group'][0].i
            w_name = node.input[1]
            w_dim = wgt_dims[w_name]
            # set dwcnv do not quantize.
            if group == w_dim[0]:
                quant_setting.dispatching_table.append(node.name, TargetPlatform.FP32)

        if no_quant_op_types:
            if node.op_type in no_quant_op_types:
                quant_setting.dispatching_table.append(node.name, TargetPlatform.FP32)

    # 3.Use different calibration algorithms to quantize the model and record the best.
    print('\nActivation calibration algorithm Searching ...')
    best_calib_algo = None
    quantized = None 
    min_error = None
    for calib_algo in [None, 'minmax', 'percentile', 'kl', 'mse', 'isotone']:
        try:
            quant_setting.quantize_activation_setting.calib_algorithm = calib_algo
            _quantized = quantize_onnx_model(
                onnx_import_file=input, 
                calib_dataloader=calib_dataloader,
                calib_steps=512, 
                input_shape=input_shape,
                device=device,
                setting=quant_setting, 
                platform=QUANT_PLATFROM,
                collate_fn=lambda x: x.to(device),
                verbose=0)
        except:
            continue
        reports = graphwise_error_analyse(graph=_quantized, 
                                running_device=device, 
                                method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
                                dataloader=calib_dataloader, 
                                collate_fn=lambda x: x.to(device), 
                                steps=512,
                                verbose=True
                                )
        sensitivity = [(op_name, error) for op_name, error in reports.items()]
        sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)
        top4error = sum([i[1] for i in sensitivity[:10]])
        if min_error is None or top4error < min_error:
            min_error = top4error
            quantized = _quantized
            best_calib_algo = calib_algo

    print('\nBest activation calibration algo:', best_calib_algo)
    # 4.layerwise analyse, Automatically set the first 4 layers with the largest quantization error to 'FP32'.
    if layerwise_analyse:
        reports = layerwise_error_analyse(graph=quantized, 
                                running_device=device, 
                                method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
                                dataloader=calib_dataloader, 
                                collate_fn=lambda x: x.to(device), 
                                steps=512)
        sensitivity = [(op_name, error) for op_name, error in reports.items()]
        sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)

        first4layer_sensitivity = [x for x in sensitivity if x[1] > 0.1]
        for op_name, _ in first4layer_sensitivity:
            quant_setting.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)

    # 5.final quant
    quant_setting.quantize_activation_setting.calib_algorithm = best_calib_algo
    if finetune:
        quant_setting.lsq_optimization = True
        quant_setting.lsq_optimization_setting.steps = 500
        quant_setting.lsq_optimization_setting.collecting_device = device
    
    quantized = quantize_onnx_model(
        onnx_import_file=input, 
        calib_dataloader=calib_dataloader,
        calib_steps=len(calib_dataloader), 
        input_shape=input_shape,
        device=device,
        setting=quant_setting, 
        platform=QUANT_PLATFROM,
        collate_fn=lambda x: x.to(device),
        verbose=1)

    # 6.final graphwise error analyse
    reports = graphwise_error_analyse(graph=quantized, 
                            running_device=device, 
                            method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
                            dataloader=calib_dataloader, 
                            collate_fn=lambda x: x.to(device), 
                            steps=len(calib_dataloader)
                            )

    print('Network quantization error calculating ...')
    # 7.collect ppq execution result for validation
    calib_dataloader_num = len(calib_dataloader)
    executor = TorchExecutor(graph=quantized, device=device)
    num_output = len(quantized.outputs)
    quant_ppq_results = []
    for sample in tqdm(calib_dataloader, desc='PPQ GENERATEING REFERENCES', total=calib_dataloader_num):
        result = executor.forward(inputs=sample.to(device))
        quant_ppq_results.append([ _.cpu().flatten() for _ in result])

    # 8.export quantized model.
    model_name = os.path.basename(input)
    quant_model_path = os.path.join(outdir, model_name.replace('.onnx', '_int8.onnx'))
    export_ppq_graph(
        graph=quantized, 
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=quant_model_path)


    # 9.calculate quantization accuracy.
    #   record input & output name
    int8_input_names  = [name for name, _ in quantized.inputs.items()]
    int8_output_names = [name for name, _ in quantized.outputs.items()]
    #   run quant model with onnxruntime.
    session = onnxruntime.InferenceSession(quant_model_path)
    quant_onnxruntime_results = []
    for sample in tqdm(calib_dataloader, desc='ONNXRUNTIME INT8 MODEL GENERATEING OUTPUTS', total=calib_dataloader_num):
        result = session.run(int8_output_names, {int8_input_names[0]: convert_any_to_numpy(sample)})
        quant_onnxruntime_results.append([convert_any_to_torch_tensor(_).flatten() for _ in result])

    # record input & output name
    fp32_input_names  = []
    fp32_output_names = []
    onrt_session = onnxruntime.InferenceSession(input)
    for iTensor in onrt_session.get_inputs():
        fp32_input_names.append(iTensor.name)
    for oTensor in onrt_session.get_outputs():
        fp32_output_names.append(oTensor.name)
    # run original model with onnxruntime.
    session = onnxruntime.InferenceSession(input)
    onnxruntime_results = []
    for sample in tqdm(calib_dataloader, desc='ONNXRUNTIME FP32 MODEL GENERATEING OUTPUTS', total=calib_dataloader_num):
        result = session.run(fp32_output_names, {fp32_input_names[0]: convert_any_to_numpy(sample)})
        onnxruntime_results.append([convert_any_to_torch_tensor(_).flatten() for _ in result])

    ppq_run_quant_cosin = []
    onrt_run_quant_cosin = []
    for onrt_res, q_ppq_res, q_onrt_res in zip(onnxruntime_results, quant_ppq_results, quant_onnxruntime_results):
        for i in range(num_output):
            ppq_run_quant_cosin.append(torch_cosine_similarity(q_ppq_res[i], onrt_res[i]))
            # onrt_run_quant_cosin.append(torch_cosine_similarity(q_onrt_res[(i+1)%4], onrt_res[i]))
            onrt_run_quant_cosin.append(torch_cosine_similarity(q_onrt_res[i], onrt_res[i]))
    ppq_run_quant_cosin = np.array(ppq_run_quant_cosin).reshape(calib_dataloader_num, num_output).transpose(1,0)
    onrt_run_quant_cosin = np.array(onrt_run_quant_cosin).reshape(calib_dataloader_num, num_output).transpose(1,0)

    ppq_run_quant_cosin_avg = [sum(ppq_run_quant_cosin[i]) / calib_dataloader_num * 100 for i in range(num_output)]
    onrt_run_quant_cosin_avg = [sum(onrt_run_quant_cosin[i]) / calib_dataloader_num * 100 for i in range(num_output)]

    ppq_run_info =  "PPQ         RUN Quant VS FP32 REF, Cosine similarity: "
    onrt_run_info = "OnnxRuntime RUN Quant VS FP32 REF, Cosine similarity: "
    for i in range(num_output):
        ppq_run_info += "{}:{} ".format(fp32_output_names[i], ppq_run_quant_cosin_avg[i])
        onrt_run_info += "{}:{} ".format(fp32_output_names[i], onrt_run_quant_cosin_avg[i])
    print(ppq_run_info)
    print(onrt_run_info)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PPQ quant onnx model tool.')

    # Positional arguments.
    parser.add_argument('--input', type=str, required=True, help='Input ONNX model')
    parser.add_argument('--shape', type=str, required=True, help='Input shape. The value should be "input0:1x3x256x256 input1:1x3x128x128"')
    parser.add_argument('--calibration', type=str, required=True, help='Path to a file specifying the trial inputs. This file should be a plain text file, containing one absolute file path per line. These files will be taken to constitute the trial set. Each path is expected to point to a image. Example: <input_tensor_name>=<input_image_path>')
    parser.add_argument('--mean', type=str, required=True, help='Image normalization parameter. The value should be "input0:v0,v1...vn input1:v0,v1...vn"')
    parser.add_argument('--std', type=str, required=True, help='Image normalization parameter. The value should be "input0:v0,v1...vn input1:v0,v1...vn"')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to store output artifacts')
    parser.add_argument('--colorformat', type=str, required=True, help='Color format, one of Gray/RGB/BGR')

    # Optional arguments.
    parser.add_argument('--no-quant-op-types', default=None, nargs='+', help='Specify partial nodes non-quantization by node type.')
    parser.add_argument('--no-quant-network-output', action='store_true', default=None, help='Whether to quantify the output of the network.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
        help='Use GPU to accelerate the quantization if the GPU environment has been properly setups')
    parser.add_argument('--layerwise_analyse', action='store_true', default=None,
        help='Measure the quantization error of each operation, it takes more computing time.')
    parser.add_argument('--finetune', action='store_true', default=None,
        help='Initiate network retraining process to reduce quantization error')
    args = parser.parse_args()

    if args.device=='cuda':
        with ENABLE_CUDA_KERNEL():
            main(args)
    else:
        with DISABLE_CUDA_KERNEL():
            main(args)