import cv2
import sys
import rclpy
import argparse
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import String

from warehouse_segment_detect_publisher.ground_segmentor import GroundSegmentor
from warehouse_segment_detect_publisher.pallet_detector import PalletDetector

class Zed2iInference(Node):
    """
    Node that subscribes to RGB and depth image topics, performs pallet detection and ground segmentation on Zed2i input,
    and publishes the annotated results.
    """

    def __init__(self, args):
        """
        Initializes the node, sets up subscribers for RGB and depth image topics.

        Launch Parameters:
            image_topic (str): RGB image topic, 'rgb/image_rect_color' by default (https://www.stereolabs.com/docs/ros/zed-node).
            depth_topic (str): Depth image topic, 'depth/depth_registered' by default (https://www.stereolabs.com/docs/ros/zed-node).
            models_path (str): path to models folder (default starts at top-level).
            optimize (bool): Whether or not to convert models to suitable for edge deployment.
        """
        super().__init__('inference_node')

        self.declare_parameter('image_topic', 'rgb/image_rect_color')
        self.declare_parameter('depth_topic', 'depth/depth_registered')
        self.declare_parameter('models_path', './models')
        self.declare_parameter('optimize', False)

        self.image_topic = args.image_topic or self.get_parameter('image_topic').get_parameter_value().string_value
        self.depth_topic = args.depth_topic or self.get_parameter('depth_topic').get_parameter_value().string_value
        self.models_path = args.models_path or self.get_parameter('models_path').get_parameter_value().string_value
        self.optimize = args.optimize if args.optimize is not None else self.get_parameter('optimize').get_parameter_value().bool_value

        self.bridge = CvBridge()

        # Subscribe to RGB image topic
        self.image_subscriber = self.create_subscription(
            Image, 
            self.image_topic, 
            self.image_callback, 
            10)

        # Subscribe to depth image topic
        self.depth_subscriber = self.create_subscription(
            Image, 
            self.depth_topic, 
            self.depth_callback, 
            10)

        self.get_logger().info(f"Subscribed to topics: {self.image_topic}, {self.depth_topic}")


    def image_callback(self, msg):
        """
        RGB image callback.
        Performs pallet detection on the received image.

        Args:
            msg (sensor_msgs.msg.Image): Incoming image message.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detector = PalletDetector(cv_image, self.models_path, self.optimize)
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
            segmentor = GroundSegmentor(cv_depth, self.models_path, self.optimize)
            segmentor.segment()
        except Exception as e:
            self.get_logger().error('Depth callback error.')


def main(args=None):

    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Run Zed2i inference node with optional args.")
    
    parser.add_argument(
        '--image_topic',
        type=str,
        default='rgb/image_rect_color',
        help='ROS topic name for RGB image (default: rgb/image_rect_color)'
    )

    parser.add_argument(
        '--depth_topic',
        type=str,
        default='depth/depth_registered',
        help='ROS topic name for depth image (default: depth/depth_registered)'
    )

    parser.add_argument(
        '--models_path',
        type=str,
        default='./models',
        help='Path to folder containing trained models (default: ./models)'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable model optimization (TensorRT conversion for edge deployment)'
    )

    params = parser.parse_args()

    inference = Zed2iInference(params)

    rclpy.spin(inference)
    rclpy.shutdown()

if __name__ == "__main__":
    main()