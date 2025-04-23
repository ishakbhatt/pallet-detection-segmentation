import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/isha/repos/pallet-detection-segmentation/ros2_ws/install/warehouse_segment_detect_publisher'
