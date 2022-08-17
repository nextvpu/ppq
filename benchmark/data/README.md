

# Dirctory structure
* imgnet:  For Classification models  
  - val2012: imageNet_val.sh and images
* coco:  For Detection models    
  - annotations: instances_val2017.json  
  - val2017: images
* cityscapes:  For Segmentation models  
  - gtFine/val  
  - leftImg8bit/val


# Dataset preprocessing
* imgnet:  Imgnet2012, ILSVRC2012_img_val.tar  
    download from:  
    Need to add labes as classification.  
    You can refer to https://blog.csdn.net/qq_33328642/article/details/122073200  
    We provide this script 'imageNet_val.sh' in ./imgnet/val2012  
    Step as:  
    - Decompress ILSVRC2012_img_val.tar  
    - Put images into ./imagenet/val2012  
    - cd ./imagenet/val2012  
    - ./imageNet_val.sh  
* coco:  COCO2017, val2017.zip  
    download from:  
    We provide this script instances_val2017.json in ./coco/annotations  
    You just need to decompress val2017.zip.  
* cityscapes:  Cityscapes  
    download from:   
    Prepare gtFine and leftImg8bit sub dataset, and decompress them.
    Then the directory is :  
    - gtFine/val/frankfurt  
    - gtFine/val/lindau  
    - gtFine/val/munster  
    - leftImg8bit/val/frankfurt  
    - leftImg8bit/val/lindau  
    - leftImg8bit/val/munster  
    You must create 'xxxxlableTrainIds.png'.  
    Yon can use this project: https://github.com/mcordts/cityscapesScripts.  
    Download cityscapesScripts-master and then excute createTrainIdLabelImgs.py in ./cityscapesScripts-master/cityscapesscripts/preparation  
    Just do as description of cityscapesScripts-master.





    
    