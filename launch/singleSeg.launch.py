from launch import LaunchDescription
from launch_ros.actions import Node

def generateLaunchDescription():
    return LaunchDescription([
        Node(
            package = 'myPerceptionPkg',
            executable = 'singleSegNode',
            name = 'singleSegNode',
            parameters = [{
                'enginePath': 'changeme/singleSegModel.engine'
            }]
        )
    ])