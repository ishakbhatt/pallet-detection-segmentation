import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from warehouse_segment_detect_publisher.cv2_to_trt_input import cv2_to_trt_input
from ultralytics import YOLO
import cv2
import os


class GroundSegmentor(Node):
    """
    Node that performs ground segmentation using previously trained model.
    Takes a depth image as input and publishes an annotated segmentation result.
    """

    def __init__(self, depth_image, models_path, optimize):
        """
        Initializes the node, loads the YOLO model, and prepares the publisher and input.

        Args:
            depth_image (np.ndarray): Depth image (as a NumPy array) used for segmentation.
        """
        super().__init__('ground_segmentor')
        self.model = YOLO(os.path.join(models_path, 'segmentation', 'best.pt')) 
        self.depth_image = depth_image 
        self.bridge = CvBridge() 
        self.models_path = models_path
        self.optimize = optimize
        self.ground_segment_publisher_ = self.create_publisher(Image, 'ground_segmentation/image', 10)


    def segment(self):
        """
        Runs inference on depth image, overlays the segmentation results,
        and publishes the annotated image to ground_segmentation/image.
        """
        # broaden variable scope
        results = 0 

        if self.optimize:
            #tensorrt_model = YOLO(os.path.join(models_path, 'segmentation', 'best.engine'))
            #tensorrt_input = cv2_to_trt_input(self.depth_image)
            #results = tensorrt_model(tensorrt_input)

            # TODO: TensorRT optimization needs further investigation to integrate for inference
            trt_inference = TensorRTInference(os.path.join(models_path, 'detection', 'best.engine'))
            results = trt_inference.infer(self.image)
            
        else:
            results = self.model(self.depth_image)  
        
        annotated_frame = results[0].plot()  

        ros_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')  
        self.ground_segment_publisher_.publish(ros_msg) 
        self.get_logger().info('Ground segmented.')
