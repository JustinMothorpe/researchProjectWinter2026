
1. Purpose
this tutorial should help you through setting up the full software environment required to run the multitask perception system, build the TensorRT engines, and execute benchmarking scripts on the Jetson Orin NX

2. Prerequisites
hardware:
    Jetson Orin NX(Yaboom or NVIDIA)
    USB Camera or compatible ribbon camera (the yahboom jetson CANNOT use the JBM228, this is setup with a usb camera currently, so there will be a section on modifying scripts when applicable)
software:
    JetPack 6.X (Ubuntu 22.04 + CUDA + TensorRT)
    Python 3.8 through 3.10
    pip
    Git

3. Expected Output
By the end of this tutorial, you should have:
    A working Python environment
    All required dependencies installed
    Verified CUDA * TensorRT functionality.

4. Instructions
4.1. open command interface
open the command interface, if one of these commands causes reboot, you will need to open it again
4.2. Update system
run:
    sudo apt update && sudo apt upgrade -y
4.3. install python tools
run:
    sudo apt install python3-pip python3-dev -y
4.4. install required python packages
run:
    pip3 install numpy pyyaml
4.5. Verify CUDA
run:
    nvcc --version

you might see something like: 
jetson@yahboom:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:53:06_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
jetson@yahboom:~$ 

4.6. Verify TensorRT
run:
python3 - << 'EOF'
import tensorrt as trt
print("TensorRT version:", trt.__version__)
EOF
4.7. Verification
you should see something like:
    TensorRT version: 10.x.x
4.8. common errors and fixes
-TensorRT not found
you'r likely using the wrong Python version
Check:
    python -c "import sys: print(sys.executatble)"
it must be:
    /usr/bin/python3
-pip installs conflicting with opencv
If you accidentally install pip OpenCV:
    pip3 uninstall opencv-python
JetPack's OpenCV is already optemized for the Jetson GPU.

7. Next Steps
02_building_engines.md
