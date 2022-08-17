
# How to 
* Prepare data, please see ./data/README.md
* Prepare models, please see ./models/README.md
* All source codes and scripts in ./src

* If you want to test Classification, please excute ./evaluation_with_imagenet.py  
* If you want to test Detection and Segmentation, please excute ./ProgramEntrance.py  
* If you want to convert model from PyTorch to ONNX just for fun, try to ./export_onnx.py  
  However, if you are a master, you can ignore it and use your own method.  


# evaluation_with_imagenet.py
* Some meaningful advice  
    - CFG_NVP: We must care about 'CFG_NVP = True'. If you choose nvp pass, please set defualt True, if not , set False.  
    - CFG_VALIDATION_DIR：Set validation dataset.  
    - CFG_TRAIN_DIR: Set calibrate dataset.  
    - dispatcher: Suggest 'conservative'.  
    - Function 'evaluate_onnx_module_with_imagenet' for running quantized models on onnxrutime.    
    - Output: For output files or models, setup as needed.   
* Run:  
  ./evaluation_with_imagenet.py


# ProgramEntrance.py
* Some meaningful advice  
    - CFG_NVP: We must care about 'CFG_NVP = True'. If you choose nvp pass, please set defualt True, if not , set False.  
    - CFG_MODELS: You can add or delete test models as you need.  
    - CFG_REQUIRES: This is a useful controller.  
        - 'ERROR_ANALYSE': 'True' means open 'graphwise_error_analyse ' for analysing accurary.  
        - 'EVALUATION' ：'True' means open accurary benchmark.  
    - dispatcher: Suggest 'pplnn' for Detection, 'conservative' for Segmentation.  
    - Modify data or modale directory as you need.  
      We use data:test as default.   
        - detection.py: in ./Utilities/ParrotsPrimary/configs/  
            - data_root: where is your coco.  
            - data: test: where is your ann_file and img_prefix.  
        - segmentation.py: in ./Utilities/ParrotsPrimary/configs/  
            - data_root: where is your cityscapes.  
            - data: test: where is your img_dir and ann_dir.  
* Run:  
  ./ProgramEntrance.py  

# Summary  
  If you follow the above steps correctly, and there is no problem, congratulations, my friend!  
  If not, please analyze it carefully by yourself!

