# Pallet Detection and Ground Segmentation (ROS 2)

Real-time ROS 2 package for detecting pallets and performing ground segmentation using ZED2i camera input. This repository is designed for warehouse automation scenarios and is deployable on edge devices like NVIDIA Jetson Orin with optional model optimization via TensorRT. 

## Features
- **Pallet Detection**: Object detection model fine-tuned on YOLOv11 (nano) to detect Euro and GMA pallets.
- **Ground Segmentation**: Segmentation model fine-tuned on YOLOv11-seg (small) to segment ground surfaces.
- **ROS2 Node**: ROS2 node to detect pallets the ground from a Zed2i camera.
- Configurable via command-line arguments or ROS 2 launch files
- **Edge Optimization**: When `true` is passed to `optimize`, node converts detection and segmentation models to formats suitable for edge deployment ([TensorRT](https://github.com/NVIDIA/TensorRT)).


## Annotation
- **Pallet Detection**: Data was automatically labeled using DINO as the backend in Roboflow. The annotations were checked manually before the labeled data was used for training.
- **Ground Segmentation**: 150 images were manually labeled in Roboflow, Yolov11-seg (small) was fine-tuned on this dataset, the rest were automatically labeled via inference using [this script](https://github.com/ishakbhatt/pallet-detection-segmentation/blob/master/segmentation/annotate.py), and they were all manually checked and adjusted in Roboflow before training the model with the full dataset.

## Training 
- **Pallet Detection**: YOLOv11 (nano) trained on 35k images after augmentation. The original dataset contains data from a [pallets dataset](https://drive.google.com/drive/folders/1xSqKa55QrNGufLRQZAbp0KFGYr9ecqgT) and the remaining images are from a previously annotated dataset.
- **Ground Segmentation**: YOLOv11-seg (small) trained on ~1500 images after augmentation. The original dataset is the pallets dataset used for detection.
- **Data Augmentation**: Training data can be augmented through the [ImageAugmentation class](https://github.com/ishakbhatt/pallet-detection-segmentation/blob/master/utils/augmentation.py) through lighting and coloring changes. The training data is then shuffled to prevent overfitting due to images showing up in a pattern.
- Trained on NVIDIA GeForce RTX 3060 GPU

## Performance Evaluation

## Requirements
- ROS 2 Humble
- Python 3.10+
- ZED SDK (for ZED2i camera)


## Installation
Clone the repo and build the ROS 2 workspace:

```bash
cd ~/repos
git clone https://github.com/ishakbhatt/pallet-detection-segmentation.git
cd pallet-detection-segmentation/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon 
source install/setup.bash
```

## Running the Node
### With Command Line Arguments (No Launch File):
```bash
ros2 run warehouse_segment_detect_publisher inference \
  --ros-args \
  -p image_topic:=rgb/image_rect_color \
  -p depth_topic:=depth/depth_registered \
  -p models_path:=./models \
  -p optimize:=true
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


## Directory Structure
```
ros2_ws/
├── src/
│   └── warehouse_segment_detect_publisher/
│       ├── inference.py
│       ├── ground_segmentor.py
│       ├── pallet_detector.py
│       ├── ...
├── launch/
│   └── warehouse_segment_detect.launch.py
```

## License
Apache 2.0

## Maintainer
Isha Bhatt  
Robotics & ML Engineer | ishakbhatt25[at]gmail[dot]com

