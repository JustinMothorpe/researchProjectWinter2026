from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from laungh.action import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'enginePath',
            default_value='engine/singleDepth.engine',
            description='Path to depth TensorRT engine'
        ),

        Node(
            package = 'myPerceptionPkg',
            executable = 'singleDepthNode',
            name = 'singleDepthNode',
            parameters = [{
                'enginePath': 'changeme/singleDepthModel.engine'
            }]
        )
    ])