import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os


class GroundSegmentor(Node):
    """
    Node that performs ground segmentation using previously trained model.
    Takes a depth image as input and publishes an annotated segmentation result.
    """

    def __init__(self, depth_image, models_path):
        """
        Initializes the node, loads the YOLO model, and prepares the publisher and input.

        Args:
            depth_image (np.ndarray): Depth image (as a NumPy array) used for segmentation.
        """
        super().__init__('ground_segmentor_node')
        self.model = YOLO(os.path.join(models_path, 'segmentation', 'best.pt')) 
        self.depth_image = depth_image  # Input depth image
        self.bridge = CvBridge() 

        # Publisher for the segmented output image
        self.ground_segment_publisher_ = self.create_publisher(Image, 'ground_segmentation/image', 10)


    def segment(self):
        """
        Runs inference on depth image, overlays the segmentation results,
        and publishes the annotated image to ground_segmentation/image.
        """
        results = self.model(self.depth_image)  # Run segmentation model
        annotated_frame = results[0].plot()  # Draw segmentation results on the image

        ros_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')  # Convert to ROS image message
        self.ground_segment_publisher_.publish(ros_msg)  # Publish the annotated image
        self.get_logger().info('Ground segmented.')
