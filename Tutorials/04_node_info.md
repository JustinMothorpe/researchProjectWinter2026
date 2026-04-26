1. Purpose
this file explains now to run the TensorRT engine nodes using the provided ROS2 launch files and how to switch between USB and ribbon (aka CSI) cameras on the Jetson Orin NX. it also covers how to set engine paths and verify node outputs
2. Prerequisites
    Jetson Orin NX with Jetpack installed
    ROS2 Installed and sourced
    your package built with:
        colcon build
        source install/setup.bash
    TensorRT engines built
    a working camera:
        usb webcam 
        or
        CSI Ribbon camera
    launch files contained in launch/
    project files and their location on the device referenced as [project] 
3. Expected output
After completeing this tutorial, you will be able to:
    launch any inference node within this project using a ROS2 Launch file
    Switch between USB and CSI cameras
    View inference node results on ROS2 topics such as: 
        /det/image
        /seg/image
        /depth/image
        /multitask/detImage
        /multitask/segImage
        /multitask/depthImage
4. Instructions
4.1. Set the Engine Path in Launch Files
Each launch file contains:
    parameters=[(
        'enginePath':  'changeme/...engine'
    )]
    Replace "changeme/...engine" with "[project]/...engine"
    this engures the node loads the correct TensorRT engine.
4.2 Running the ROS2 Nodes
MultitaskNode:
    ros2 launch myPerceptionPkg multiTask.Launch.py
Single-Task Nodes:
    Detection:
        ros2 launch myPerceptionPkg singleDet.Launch.py
    Segmentation:
        ros2 launch myPerceptionPkg singleSeg.Launch.py
    Depth
        ros2 launch myPerceptionPkg singleDepth.Launch.py

4.3 Switching Between USB and Ribbon Cameras
All nodes subscribe to /camera/image_raw
So whichever camera you use must publish to that topic.

USB Camera (UVC)
start the USB camera node with:
    ros2 run v412-camera --ros-args -r image-raw:=/camera/image_raw
this remaps the default /image_raw topic to /camera/image_raw

Ribbon/CSI Camera
Start the CSI Camera node with:
    ros2 run csi_camera csi_camera_node
if needed, remap:
    ros2 run csi_camera csi_camera_node --ros-args -r image-raw:=/camera/image_raw
if the CSI camera Fails:
    sudo systemctl restart nvargus-daemon

4.4 Viewing Node Output
use the command:
    ros2 run rqt_image_view rqt_image_view
select:
    /det/image
    /seg/image
    /depth/image
    /multitask/detImage
    /multitask/segImage
    /multitask/depthImage

5. Verification
Check the camera topic:
    ros2 topic list
you should see 
    /camera/image_raw

check output node topics
    ros2 topic list
you should see one or more of:
    /det/image
    /seg/image
    /depth/image
    /multitask/detImage
    /multitask/segImage
    /multitask/depthImage

Check engine path parameter
    ros2 param get /multiTaskNode enginePath
6. Common Errors & Fixes
Error: Node recieves no images
FIx:
    -r image_raw:=/camera/image_raw

Error: CSI camera fails to start
Fix:
    sudo systemctl restart nvargus-daemon

Error: Engine path incorrect
Fix the launch file


7. Next Steps
proceed to 05_troubleshooting.md