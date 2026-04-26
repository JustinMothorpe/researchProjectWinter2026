1. Purpose
Provide solutions to the most common issues encountered during setup, engine building, inference, and benchmarking.

2. Common Issues
--Error: Camera Not Detected
Fix:
ls /dev/video*

if there is something 
use the /dev/video# that corrosponds
if not
there is either: 
    no ribbon connection
    no reboot after a camera is connected
    you are using a usb camera

if you are using a ribbon camera:
    first check that the camera is seated properly.
    then reboot the sbc
    refer to the appropriate parts in 04_node_info

if you are using a usb camera:
    refer to the appropriate parts in 04_node_info

--Error: TensorRT Engine Fails to Build
fix:
    Reduce workspace
    re-export to onnx
    ensure FP16 support

--Error: Benchmark Script Crashes
fix: ensure input shape matches engine and that engines are built.

--Error: Low FPS
    Switch to 15W or 25W mode:
sudo nvpmodel -m 2
sudo jetson_clocks

