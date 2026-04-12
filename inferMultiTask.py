import numpy as np
from trtInferUtils import TRTInference
from .src.sharedUtils.sizeMain import MainSize

def main():
    size = MainSize(480, 640)
    trtModel= TRTInference("multiTastModel.engine")

    img = np.random.randn(1, 3, size.getSize(False), size.getSize(True)).astype(np.float32)

    outputs = trtModel.infer(img)

    print("detection:", outputs["detection"].shape)
    print("segmentation:", outputs["segmentation"].shape)
    print("depth:", outputs["depth"].shape)

if __name__ == "__main__":
    main()