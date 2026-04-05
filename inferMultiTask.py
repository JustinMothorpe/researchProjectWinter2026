import numpy as np
from trtInferUtils import TRTInference

def main():
    trtModel= TRTInference("multiTastModel.engine")

    img = np.random.randn(1, 3, 480, 640).astype(np.float32)

    outputs = trtModel.infer(img)

    print("detection:", outputs["detection"].shape)
    print("segmentation:", outputs["segmentation"].shape)
    print("depth:", outputs["depth"].shape)

if __name__ == "__main__":
    main()