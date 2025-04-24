import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String 
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os

from warehouse_segment_detect_publisher.tensorrt_inference import TensorRTInference


class PalletDetector(Node):
    """
    Node that performs pallet detection using a YOLO segmentation model.
    Subscribes to an image input and publishes the detection results as annotated images.
    """

    def __init__(self, image, models_path, optimize):
        """
        Initializes the node, loads the model, and sets up pallet detector node.

        Args:
            image (np.ndarray): Input image as a NumPy array to run inference on.
            optimize (bool): Whether or not to convert models to suitable for edge deployment.
        """
        super().__init__('pallet_detector')
        self.model = YOLO(os.path.join(models_path, 'detection', 'best.pt'))  
        self.image = image  
        self.bridge = CvBridge()  
        self.optimize = optimize
        self.pallet_detector_publisher_ = self.create_publisher(Image, 'pallet_detector/image', 10)

    def detect(self):
        """
        Runs inference on the input image, overlays the detection results, and publishes it.
        """
        results = 0 

        if self.optimize:
            #tensorrt_model = YOLO(os.path.join(models_path, 'detection', 'best.engine'))
            #tensorrt_input = cv2_to_trt_input(self.image)
            #results = tensorrt_model(tensorrt_input)
            # TODO: TensorRT optimization needs further investigation to integrate for inference
            trt_inference = TensorRTInference(os.path.join(models_path, 'detection', 'best.engine'))
            results = trt_inference.infer(self.image)
        else:
            results = self.model(self.image)  
            
        annotated_frame = results[0].plot() 

        ros_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')  
        self.pallet_detector_publisher_.publish(ros_msg)  

        self.get_logger().info('Pallets Detected.') 
