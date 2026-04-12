import numpy as np
from trtInferUtils import TRTInference
from .src.sharedUtils.sizeMain import MainSize

def main():
    size = MainSize(480, 640)
    trtModel = TRTInference("singleDepth.engine")
    
    img = np.random.randn(
        1, 
        3, 
        size.getSize(False), 
        size.getSize(True)
    ).astypes(np.float32)

    outputs = trtModel.infer(img)

    print("depth:", outputs["depth"].shape)
    
if __name__ == "__main__":
    main()