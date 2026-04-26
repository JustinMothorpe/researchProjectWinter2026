from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from laungh.action import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'enginePath',
            default_value='engine/multiTaskModel.engine',
            description='Path to mutlitask TensorRT engine'
        ),

        Node(
            package = 'myPerceptionPkg',
            executable = 'multiTaskNode',
            name = 'multiTaskNode',
            parameters = [{
                'enginePath': 'changeme/multiTaskModel.engine'
            }]
        )
    ])