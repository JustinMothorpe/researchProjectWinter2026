import cv2
import numpy as np

def preprocess(frame):
    frameResized = cv2.resize(
        frame, 
        (
            640, 
            480
        )
    )
    img = frameResized.astype(np.float32) / 255.0
    img = img.transpose(
        2, 
        0, 
        1
    ) 
    return img[np.newaxis, :, :, :]

def overlaySegmentation(frame, seg):
    segMask = (seg[0,0] > 0.5).astype(np.uint8) *  255
    segMask = cv2.resize(
        segMask, (
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

def depthToColorMap(depth):
    depthNorm = cv2.normalize(
        depth[
            0, 
            0
        ], 
        None, 
        0, 
        255, 
        cv2.NORM_MINMAX
    )
    depthUint8 = depthNorm.astype(np.uint8)
    return cv2.applyColorMap(
        depthUint8, 
        cv2.COLORMAP_MAGMA
    )

def drawDetections(frame, det):
    det = det[0]
    h, w = frame.shape[:2]

    for box in det:
        x1, y1, x2, y2, score, cls = box
        if score < 0.5:
            continue

        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        cv2.rectangle(
            frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (
                0,
                255,
                0
            ),
            2
        )
        cv2.putText(
            frame,
            f"{score:.2f}",
            (
                x1,
                y1-5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                0
            ),
            2
        )
    
    return frame

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


