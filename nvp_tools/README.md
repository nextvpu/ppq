## 安装

* 安装依赖，请参考[README](https://github.com/nextvpu/ppq/blob/master/README.md)

* 从源码安装PPQ:

1. Run following code with your terminal(For windows user, use command line instead).

```bash
git clone https://github.com/nextvpu/ppq.git
git checkout -b dev origin/dev
cd ppq
pip install -r requirements.txt
python setup.py install
```
## 开始量化

* 准备量化校准数据集文件，文件为纯文本文件，每行包含一个图片文件绝对路径。例如：

        <input_tensor_name>=<input_image_path_1>
        <input_tensor_name>=<input_image_path_2>
        <input_tensor_name>=<input_image_path_3>
                        ......
        <input_tensor_name>=<input_image_path_n>

* 查看命令行参数及帮助信息

```bash
cd nvp_tools
python PPQ_quant_tool.py --help
```

* 命令行执行

```bash
python PPQ_quant_tool.py --model-type <onnx/caffe> --model <Path to fp32 onnx model/ caffemodel> --proto <Path to caffe prototxt> --shape <Input shape> --calibration <Path to calibration file> --mean <Image normalization parameter, mean>  --std <Image normalization parameter, std> --colorformat <Color format Gray/RGB/BGR> --outdir <Directory to store output artifacts> --device <cpu/cuda> --chip-type <n16x/n161sx>
```

## 测试sample

* 命令行执行

```bash
cd nvp_tools
python PPQ_quant_tool.py --model-type onnx --model ./sample/test_case.onnx --shape data:1x1x112x112 --calibration ./sample/calib.txt --mean data:128  --std data:128 --colorformat Gray --outdir ./sample --device cuda  --chip-type n16x
```