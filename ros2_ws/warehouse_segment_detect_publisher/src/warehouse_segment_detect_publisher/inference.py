import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import String
import cv2

from warehouse_segment_detect_publisher.ground_segmentor import GroundSegmentor
from warehouse_segment_detect_publisher.pallet_detector import PalletDetector

class Zed2iInference(Node):
    """
    Node that subscribes to RGB and depth image topics, performs pallet detection and ground segmentation on Zed2i input,
    and publishes the annotated results.
    """

    def __init__(self, image_topic='rgb/image_rect_color', depth_topic='depth/depth_registered', models_path='./models'):
        """
        Initializes the node, sets up subscribers for RGB and depth image topics.

        Args:
            image_topic (str): RGB image topic, 'rgb/image_rect_color' by default (https://www.stereolabs.com/docs/ros/zed-node).
            depth_topic (str): Depth image topic, 'depth/depth_registered' by default (https://www.stereolabs.com/docs/ros/zed-node).
            models_path (str): path to models folder (default starts at top-level).
        """
        super().__init__('inference_node')

        self.models_path = models_path
        self.bridge = CvBridge()

        # Subscribe to RGB image topic
        self.image_subscriber = self.create_subscription(
            Image, 
            image_topic, 
            self.image_callback, 
            10)

        # Subscribe to depth image topic
        self.depth_subscriber = self.create_subscription(
            Image, 
            depth_topic, 
            self.depth_callback, 
            10)

        self.get_logger().info(f"Subscribed to topics: {image_topic}, {depth_topic}")


    def image_callback(self, msg):
        """
        RGB image callback.
        Performs pallet detection on the received image.

        Args:
            msg (sensor_msgs.msg.Image): Incoming image message.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detector = PalletDetector(cv_image, self.models_path)
            detector.bridge = self.bridge
            detector.detect()
        except Exception as e:
            self.get_logger().error('image callback error.')


    def depth_callback(self, msg):
        """
        Depth image callback.
        Performs ground segmentation on the received depth image.

        Args:
            msg (sensor_msgs.msg.Image): Incoming depth image message.
        """
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            segmentor = GroundSegmentor(cv_depth, self.models_path)
            segmentor.segment()
        except Exception as e:
            self.get_logger().error('Depth callback error.')
