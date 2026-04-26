1. Purpose
This tutorial explains how to build TensorRT engines for all tasks: detection, segmentation, depth and the multitask model.
2. Prerequisites
Completed Environment Setup
downloaded project's location (referenced as [Project])
3. Expected output
you will generate.engine files such as:
    multiTaskModel.engine
    singleDet.engine
    singleSeg.engine
    singleDepth.engine
4. Instructions
4.1 
open the command line
4.2
run the following commands:
command 1:
/usr/src/tensorrt/bin/trtexec \
  --onnx=[Project]/singleDet.onnx \
  --saveEngine=[Project]/singleDet.engine \
  --fp16

command 2;
/usr/src/tensorrt/bin/trtexec \
  --onnx=[Project]/singleSeg.onnx \
  --saveEngine=[Project]/singleSeg.engine \
  --fp16
command 3:
/usr/src/tensorrt/bin/trtexec \
  --onnx=[Project]/singleDepth.onnx \
  --saveEngine=[Project]/singleDepth.engine \
  --fp16
command 4:
/usr/src/tensorrt/bin/trtexec \
  --onnx=[Project]/multiTaskModel.onnx \
  --saveEngine=[Project]/multiTaskModel.engine \
  --fp16

these build the detection, segmentation, depth, and multitask engines

5. Verification
list the generated engines:
    cd [Project]
    ls *.engine
You should see:
singleDet.engine
singleSeg.engine
singleDepth.engine
multiTaskModel.engine

6. Common errors and fixes
Error: Could not read external data file
Cause: .onnx.data file missing or moved
Fix: ensure both files are present (for example with detection):
singleDet.onnx
singleDet.onnx.data

Error: Out of memory during engine build
Fix: reduce workspace:
--workspace=2038

Error: Unsupported ONNX operations
fix: Re-export ONNX with opset 17 or Higher

7. Next Steps
proceed to 03_benchmarking.md