import cv2
import numpy as np
import time
from trtInferUtils import TRTInference
from src.sharedUtils.preprocess import preprocess, depthToColorMap, overlaySegmentation

ENGINEPATH = "multiTaskModel.engine"


def main():
    trtModel = TRTInference(ENGINEPATH)

    pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Camera not found")
        return

    start = time.time()
    prev = start
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        inp = preprocess(frame=frame)
        outputs = trtModel.infer(inp)

        #Extract outputs
        det = outputs.get("detection")
        seg = outputs.get("segmention")
        depth = outputs.get("depth")

        # Visualization
        vis = frame.copy()

        if seg is not None :
            vis = overlaySegmentation(vis, seg)
        if depth is not None :
            depthMap = depthToColorMap(depth)
            depthMap = cv2.resize(depthMap, (320, 240))
            vis[0:240, 0:320] = depthMap
        
        
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - prev))
        prev = now
        cv2.putText(
            vis, 
            f"FPS:  {fps:.1f}", 
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
            "MultiTask TensorRT Demo",
            vis
        )

        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
