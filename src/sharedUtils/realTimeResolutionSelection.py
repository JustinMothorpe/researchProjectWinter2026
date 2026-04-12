import cv2

def selectResolution(camValue):
    #video0
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    print(f"Camera running at {width}x{height}")
    return width, height