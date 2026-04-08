import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

from ....trtInferUtils import TRTInference
from ...sharedUtils.preprocess import preprocess, depthToColorMap, drawDetections, overlaySegmentation

class MultiTaskNode(Node):
    def __init__(self):
        super.__init__('MultiTaskNode')
        self.declareParameter(
            'enginePath', 
            'MultiTaskModel.engine'
        )
        enginePath = self.get_parameter('enginePath').get_parameter_value().string_value

        self.trtModel = TRTInference(enginePath)
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, 
            '/camera/image_raw',
            self.image_cb,
            10 
        )
        self.pubDet = self.create_publisher(
            Image,
            '/multitask/detImage',
            10
        )
        self.pubSeg = self.create_publisher(
            Image,
            '/multitask/segImage',
            10
        )
        self.pubDepth = self.create_publisher(
            Image,
            '/multitask/depthImage',
            10
        )

    def imageCb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        inp, base = preprocess(frame)
        outputs = self.trtModel.infer(inp)
        
        det = outputs['detection']
        seg = outputs['segmentation']
        depth = outputs['depth']

        detImg = drawDetections(base.copy(), det)
        segImg = overlaySegmentation(base.copy(), seg)
        depthMap = depthToColorMap(depth)
        depthMap = cv2.resize(
            depthMap,
            (
                base.shape[1],
                base.shape[0]
            )
        )

        stamp = self.get_clock().now().to_msg()
        frameId = msg.header.frame_id

        detMsg = self.bridge.cv2_to_imgmsg(detImg, encoding = 'bgr8')
        detMsg.header = Header(stamp = stamp, frame_id = frameId)
        self.pubDet.publish(detMsg)
        
        segMsg = self.bridge.cv2_to_imgmsg(segImg, encoding = 'bgr8')
        segMsg.header = Header(stamp = stamp, frame_id = frameId)
        self.pubSeg.publish(segMsg)

        depthMsg = self.bridge.cv2_to_imgmsg(depthMap, encoding = 'bgr8')
        depthMsg.header = Header(stamp = stamp, frame_id = frameId)
        self.pubDepth.publish(depthMsg)
        

def main(args = None):
    rclpy.init(args=args)
    node = MultiTaskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()