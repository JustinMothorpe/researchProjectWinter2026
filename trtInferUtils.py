import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TrtLogger = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, enginePath):
        self.engine = self.loadEngine(enginePath)
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.allocateBuffers()
    
    def loadEngine(self, path):
        with open(path, "rb") as f, trt.Runtime(TrtLogger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocateBuffers(self):
        numTensors = self.engine.num_io_tensors
        self.bindings = [None] * numTensors
        self.inputs = []
        self.outputs = []

        for i in range(numTensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = trt.volume(shape)
            hostMem = np.empty(size, dtype=dtype)
            deviceMem = cuda.mem_alloc(hostMem.nbytes)

            self.bindings[i] = int(deviceMem)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((hostMem, deviceMem, name, shape))
            else:
                self.outputs.append((hostMem, deviceMem, name, shape))
    
    def infer(self, input_tensor: np.ndarray):
        hostMem, deviceMem, _, _ = self.inputs[0]

        np.copyto(hostMem, input_tensor.ravel())
        cuda.memcpy_hotd(deviceMem, hostMem)

        self.context.execute_v2(self.bindings)

        results = {}
        for hostMem, deviceMem, name, shape in self.outputs:
            cuda.memcpy_dtoh(hostMem, deviceMem)
            results[name] = hostMem.reshape(shape)
        
        return results