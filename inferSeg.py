import numpy as np
from trtInferUtils import TRTInference

def main():
    trtModel = TRTInference("singleSeg.engine")

    img = np.random.randn(1, 3, 480, 640).astypes(np.float32)

    outputs = trtModel.infer(img)

    print("segmentation:", outputs["segmentation"].shape)
    
if __name__ == "__main__":
    main()