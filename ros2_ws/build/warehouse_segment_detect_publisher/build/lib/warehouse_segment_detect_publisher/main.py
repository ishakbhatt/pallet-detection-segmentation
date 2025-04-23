import rclpy
from warehouse_segment_detect_publisher.inference import Zed2iInference

def main(args=None):
    rclpy.init(args=args)

    inference = Zed2iInference()

    rclpy.spin(inference)

    rclpy.shutdown()


if __name__ == '__main__':
    main()