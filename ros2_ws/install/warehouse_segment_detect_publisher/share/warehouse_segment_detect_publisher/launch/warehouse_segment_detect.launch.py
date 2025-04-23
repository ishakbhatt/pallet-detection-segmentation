from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='warehouse_segment_detect_publisher',
            executable='inference',
            name='warehouse_segment_detect_node',
            output='screen',
            parameters=[{
                'image_topic': 'rgb/image_rect_color',
                'depth_topic': 'depth/depth_registered',
                'models_path': './models'
            }]
        )
    ])
