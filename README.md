# Pallet Detection and Ground Segmentation (ROS 2)

Real-time ROS 2 package for detecting pallets and performing ground segmentation using ZED2i camera input. This repository is designed for warehouse automation scenarios and is deployable on edge devices like NVIDIA Jetson Orin with optional model optimization via TensorRT. 

## Features
- **Pallet Detection**: Object detection model fine-tuned on YOLOv11 (nano) to detect Euro and GMA pallets.
- **Ground Segmentation**: Segmentation model fine-tuned on YOLOv11-seg (small) to segment ground surfaces.
- **ROS2 Node**: ROS2 node to detect pallets the ground from a Zed2i camera.
- Configurable via command-line arguments or ROS 2 launch files
- **Edge Optimization**: When `true` is passed to `optimize`, node converts detection and segmentation models to formats suitable for edge deployment ([TensorRT](https://github.com/NVIDIA/TensorRT)).

## Datasets
- **Pallets Detection**: The full dataset including the provided data and openly available data was combined and is available [here](https://universe.roboflow.com/objectdetector-fo7et/isha-s-pallets-dataset-detection).
- **Ground Segmentation**: The [original dataset](https://drive.google.com/drive/folders/1xSqKa55QrNGufLRQZAbp0KFGYr9ecqgT) was used.

## Annotation
- **Pallet Detection**: Data was automatically labeled using DINO as the backend in Roboflow. The annotations were checked manually before the labeled data was used for training.
- **Ground Segmentation**: 150 images were manually labeled in Roboflow, Yolov11-seg (small) was fine-tuned on this dataset, the rest were automatically labeled via inference using [this script](https://github.com/ishakbhatt/pallet-detection-segmentation/blob/master/segmentation/annotate.py), and they were all manually checked and adjusted in Roboflow before training the model with the full dataset.

## Training 
Both models were trained with an 80/20 split between training and val/test. The data was shuffled across the split as well as within each group.

- **Pallet Detection**: YOLOv11 (nano) trained on 35k images after augmentation. The original dataset contains data from a [pallets dataset](https://drive.google.com/drive/folders/1xSqKa55QrNGufLRQZAbp0KFGYr9ecqgT) and the remaining images are from a previously annotated dataset.
- **Ground Segmentation**: YOLOv11-seg (small) trained on ~1500 images after augmentation. The original dataset is the pallets dataset used for detection.
- **Data Augmentation**: Training data can be augmented through the [ImageAugmentation class](https://github.com/ishakbhatt/pallet-detection-segmentation/blob/master/utils/augmentation.py) through lighting and coloring changes. The training data is then shuffled to prevent overfitting due to images showing up in a pattern.
- Trained on NVIDIA GeForce RTX 3060 GPU

## Performance Evaluation
- Pallet Detection:
  - Best mAP@0.5: **0.967**
  - Best mAP@0.5:0.95: **0.7709**

- Ground Segmentation:
  - Best mAP@0.5: **0.672**
  - Best mAP@0.5:0.95: **0.593**
  
The mAP@0.5 and 0.5:0.95 values are the best values found in `metrics/detection`. mAP@0.5 represents the mean Average Precision at a 0.5 intersection over union (amount of overlap between the ground truth and predicted boxes - IoU). About 97% of the boxes are correct predictions because they overlap with the ground truth by at least 50%. As far as mAP@0.5:0.95, there is a drop in the performance. This calculation averages the AP over a range of IoU thresholds instead of just at 0.5. We see a drop because as the IoU increments go up, it becomes more difficult for the ground truth and predicted boxes or masks to align that much more strictly.

## Tested on:
- ROS 2 Humble
- Python 3.10+
- CUDA 12.8

## Installation
Clone the repo and build the ROS 2 workspace:

```bash
cd ~/repos
git clone https://github.com/ishakbhatt/pallet-detection-segmentation.git
cd pallet-detection-segmentation/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

## Running the Node
### With Command Line Arguments (No Launch File):
Topic names are passed, path to all the models, and the optimize flag if using the optimized model.
```bash
ros2 run warehouse_segment_detect_publisher inference \
--image_topic robot1/zed2i/left/image_rect_color \
--depth_topic depth/depth_registered \
--models_path ../models \
--optimize
```

### With Launch File:
```bash
ros2 launch warehouse_segment_detect_publisher warehouse_segment_detect.launch.py
```
Launch file parameters can be edited in `ros2_ws/launch/warehouse_segment_detect.launch.py`.


## Parameters
| Name         | Type    | Default                 | Description                                      |
|--------------|---------|-------------------------|--------------------------------------------------|
| image_topic  | string  | rgb/image_rect_color    | ZED2i RGB topic name                             |
| depth_topic  | string  | depth/depth_registered  | ZED2i Depth topic name                           |
| models_path  | string  | ./models                | Path to directory containing model files         |
| optimize     | boolean | false                   | Enable optimization (TensorRT)             |

## Pruning & Quantization with TensorRT
The script for pruning & quantization is `convert_to_tensorrt.py`. It converts the model to `onnx` format, prunes the model, and then converts `FP32` to `FP16`. 

**Usage**: run `python3 convert_to_tensorrt.py -h` for the options:

```
Convert YOLOv8 .pt model to TensorRT .engine file

options:
  -h, --help            show this help message and exit
  --models_path MODELS_PATH
                        Path to the .pt file
  --imgsz IMGSZ         Input image size (default: 416)
  --inference_method INFERENCE_METHOD
                        detection or segmentation
```

The metrics of the original and improved model is below:

| Operation    | best.onnx                  | best.slim.onnx                  |
|--------------|----------------------------|---------------------------------|
| **Model Name** | best.onnx                  | models/detection/best.slim.onnx |
| **Model Info** | Op Set: 19 / IR Version: 9 | Op Set: 19 / IR Version: 9      |
| **IN: images** | float32: (1, 3, 416, 416)  | float16: (1, 3, 416, 416)       |
| **OUT: output0** | float32: (1, 5, 3549)    | float16: (1, 5, 3549)           |
| **Add**       | 16                         | 16                              |
| **Cast**      | 0                          | 6                               |
| **Concat**    | 23                         | 23                              |
| **Conv**      | 88                         | 88                              |
| **Div**       | 1                          | 1                               |
| **MatMul**    | 2                          | 2                               |
| **MaxPool**   | 3                          | 3                               |
| **Mul**       | 79                         | 79                              |
| **Reshape**   | 8                          | 8                               |
| **Resize**    | 2                          | 2                               |
| **Sigmoid**   | 78                         | 78                              |
| **Slice**     | 2                          | 2                               |
| **Softmax**   | 2                          | 2                               |
| **Split**     | 11                         | 11                              |
| **Sub**       | 2                          | 2                               |
| **Transpose** | 3                          | 3                               |
| **Model Size**| 10.00 MB                   | 5.04 MB                         |
| **Elapsed Time** |                        | 0.21 s                          |

Note: `onnx_to_tensorrt` may throw an error:
```
[04/24/2025-13:24:39] [TRT] [W] Unable to determine GPU memory usage: no CUDA-capable device is detected
```
It is best to use `trtexec` by building [TensorRT](https://github.com/NVIDIA/TensorRT/?tab=readme-ov-file#tensorrt-open-source-software) and running the tool in the docker container. Place the `best.slim.onnx` in the top-level folder of `TensorRT`, start the docker container, run `trtexec --onnx=best.slim.onnx --saveEngine=best.engine`, and copy it back to `models/detection/` or `models/segmentation/` depending on the task.

The `--optimize` flag currently gives a callback error. This needs to be further investigated. For now, the infrastructure is in place to convert to TensorRT and use an optimized model.

## License
Apache 2.0

## Maintainer
Isha Bhatt  
Robotics & ML Engineer | ishakbhatt25[at]gmail[dot]com

