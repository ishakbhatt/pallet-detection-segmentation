import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String  # Added for String message type
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class PalletDetector(Node):
    """
    Node that performs pallet detection using a YOLO segmentation model.
    Subscribes to an image input and publishes the detection results as annotated images.
    """

    def __init__(self, image):
        """
        Initializes the node, loads the model, and sets up pallet detector node.

        Args:
            image (np.ndarray): Input image as a NumPy array to run inference on.
        """
        super().__init__('yolo_inference_node')
        self.model = YOLO(os.path.join(models_path, 'segmentation', 'best.pt'))  
        self.image = image  
        self.bridge = CvBridge()  

        self.pallet_detector_publisher_ = self.create_publisher(Image, 'pallet_detector/image', 10)

    def detect(self):
        """
        Runs inference on the input image, overlays the detection results, and publishes it.
        """
        results = self.model(self.image)  
        annotated_frame = results[0].plot() 

        ros_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')  
        self.pallet_detector_publisher_.publish(ros_msg)  

        self.get_logger().info('Pallets Detected.') 
