import tensorRT as trt
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
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty_like(size, dtype)
            device_mem = cuda.mem_alloc_like(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append((host_mem, device_mem, name, shape))
            else:
                self.outputs.append((host_mem, device_mem, name, shape))
    
    def infer(self, input_tensor: np.ndarray):
        host_mem, device_mem, _, _ = self.inputs[0]

        np.copyto(host_mem, input_tensor.ravel())
        cuda.memcpy_hotd(device_mem, host_mem)

        self.context.execute_v2(self.bindings)

        results = {}
        for host_mem, device_mem, name, shape in self.outputs:
            cuda.memcpy_dtoh(host_mem, device_mem)
            results[name] = host_mem.reshape(shape)
        
        return results