import cv2
import numpy as np
import time
from trtInferUtils import TRTInference
from src.sharedUtils.preprocess import preprocess, depthToColorMap

ENGINEPATH = "singleDepth.engine"



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
        depth = outputs["depth"]
        
        depthMap = depthToColorMap(depth)
        depthMap = cv2.resize(
            depthMap,
            (
                frame.shape[1],
                frame.shape[0]
            )
        )

        vis = depthMap
        
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
            "Single-Task Depth",
            vis
        )
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
if  __name__ == "__main__":
    main()