import numpy as np
import cv2

def cv2_to_trt_input(image, input_width, input_height):
    """
    Converts an image from a given file path to a format suitable for TensorRT input.

    See data_preprocessing.py in the yolov3 example in TensorRT: https://github.com/NVIDIA/TensorRT/blob/master/samples/python/yolov3_onnx/data_processing.py

    Args:
        image (cv2 image): Image to preprocess.
        input_width (int): New width.
        input_height (int): New height.

    Returns:
        np.ndarray: pre-processed image for TensorRT inference.
    """
    
    resized_image = cv2.resize(image, (input_width, input_height))
    
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    normalized_image = rgb_image.astype(np.float32) / 255.0
    
    transposed_image = normalized_image.transpose((2, 0, 1))
    
    input_data = np.expand_dims(transposed_image, axis=0)

    return input_data.astype(np.float32)