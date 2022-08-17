import torchvision
import torch
import os
import onnx

CFG_DEVICE = 'cuda'
CFG_DUMP_PATH = "./model/"


input_data = torch.randn(1, 3, 224, 224)
input_names = [ "input" ]
output_names = [ "output"]

if __name__ == '__main__':
    for model_builder, model_name in (
        (torchvision.models.resnext101_32x8d, 'resnext101_32x8d'),
        
    ):
        print(f'---------------------- PPQ Quantization Test Running with {model_name} ----------------------')
        model = model_builder(pretrained=True)

        file_path = f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx'

        torch.onnx.export(model,
                            input_data ,
                            file_path,
                            verbose=False,
                            input_names=input_names,
                            output_names=output_names)
        onnx_model = onnx.load(file_path) # load onnx model
        onnx.checker.check_model(onnx_model)
