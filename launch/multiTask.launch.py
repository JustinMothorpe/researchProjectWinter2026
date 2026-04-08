from launch import LaunchDescription
from launch_ros.actions import Node

def generateLaunchDescription():
    return LaunchDescription([
        Node(
            package = 'myPerceptionPkg',
            executable = 'multiTaskNode',
            name = 'multiTaskNode',
            parameters = [{
                'enginePath': 'changeme/multiTaskModel.engine'
            }]
        )
    ])