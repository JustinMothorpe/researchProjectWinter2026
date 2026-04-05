import cv2
import numpy as np
import time
from trtInferUtils import TRTInference
from src.sharedUtils.preprocess import preprocess

ENGINEPATH = "singleSeg.engine"

def overlaySegmentation(frame, seg):
    segMask = (seg[0, 0] > 0.5).astype(np.uint8) * 255
    segMask = cv2.resize(
        segMask, 
        (
            frame.shape[1], 
            frame.shape[0]
        )
    )

    color = np.zeros_like(frame)
    color[:, :, 1] = segMask
    return cv2.addWeighted(
        frame, 
        0.7, 
        color,
        0.3,
        0
    ) 

def main():
    trtModel = TRTInference(ENGINEPATH)
    cap = cv2.VideoCapture(0)

    prev = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess(frame=frame)
        outputs = trtModel.infer(inp)
        det = outputs["segmentation"]
        
        vis = overlaySegmentation(frame.copy(), det)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - prev))
        prev = now
        cv2.putText(
            vis, 
            f"FPS: {fps:.1f}", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                0,
                255,
                0
            ),
            2
        )
        cv2.imshow(
            "Single-Task Segmentation",
            vis
        )
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
if  __name__ == "__main__":
    main()