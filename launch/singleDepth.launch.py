from launch import LaunchDescription
from launch_ros.actions import Node

def generateLaunchDescription():
    return LaunchDescription([
        Node(
            package = 'myPerceptionPkg',
            executable = 'singleDeptNode',
            name = 'singleDepthNode',
            parameters = [{
                'enginePath': 'changeme/singleDepthModel.engine'
            }]
        )
    ])