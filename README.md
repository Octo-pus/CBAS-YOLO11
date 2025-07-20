# Enhanced Small Target Detection in UAV Imagery: Introducing CBAS-YOLO11 with Adaptive Feature Fusion
Official PyTorch implementation of CBAS-YOLO11

[Enhanced Small Target Detection in UAV Imagery: Introducing CBAS-YOLO11 with Adaptive Feature Fusion]
Zhijun Gao, Yu Zhang, Kaiyun Chen, Xiliang Yin


<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Unmanned Aerial Vehicle (UAV) aerial images are increasingly utilised due to their high resolution, multi-scale capability, and flexibility. However, small target detection remains challenging due to their diminutive size, weak features, and complex backgrounds. To address these challenges, we introduce CBAS-YOLO11, an advanced model based on YOLO11s, featuring a novel C3k2-CBNL structure that integrates CBAM with a non-local attention mechanism. This structure enhances global context capture while preserving local attention, crucial for small target feature extraction. Additionally, a dual-temporal feature aggregation module, inspired by BFAM, is embedded to improve multi-scale feature fusion, aligning low-level details with high-level semantics. We also incorporate a new P2 detection layer with the ASFF method for adaptive feature fusion, dynamically weighting features from different layers to enhance small target detection. Experiments on VisDrone2019 and Almaty datasets demonstrate the effectiveness of each module, with CBAS-YOLO11 achieving superior mAP compared to mainstream models, while maintaining a lightweight profile with only 15.38M parameters and achieving 41.7 FPS, ensuring both high accuracy and real-time performance.
</details>

## DataSets

Vistrone2019
(https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip,
          https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip,
          https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip)

It is important to include a script in ultralytics/cfg/datasets/VisDrone. yaml that converts Vistrone2019 to YOLO format.

Aerial Traffic Images
(https://www.kaggle.com/datasets/cihangiryiit/aerial-traffic-images)

It is also important to include a script in ultralytics/cfg/datasets/Aerial. yaml that converts Aerial Traffic Images to YOLO format.
## Installation
`conda` virtual environment is recommended. 
```
conda create -n yolo python==3.8
conda activate yolo
pip install ultralytics
```

## Training 
```
python train.py
```

## Validation
Note that a smaller confidence threshold can be set to detect smaller objects or objects in the distance. 
```
python val.py
```


## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics).

Thanks for the great implementations! 
