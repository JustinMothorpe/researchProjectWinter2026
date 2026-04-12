import numpy as np
from trtInferUtils import TRTInference
from .src.sharedUtils.sizeMain import MainSize

def main():
    size = MainSize(480, 640)
    trtModel = TRTInference("singleSeg.engine")

    img = np.random.randn(1, 3, size.getSize(False), size.getSize(True)).astypes(np.float32)

    outputs = trtModel.infer(img)

    print("segmentation:", outputs["segmentation"].shape)
    
if __name__ == "__main__":
    main()