from launch import LaunchDescription
from launch_ros.actions import Node

def generateLaunchDescription():
    return LaunchDescription([
        Node(
            package = 'myPerceptionPkg',
            executable = 'singleDetNode',
            name = 'singleDetNode',
            parameters = [{
                'enginePath': 'changeme/singleDetModel.engine'
            }]
        )
    ])