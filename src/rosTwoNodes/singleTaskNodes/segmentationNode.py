import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

from ....trtInferUtils import TRTInference
from ...sharedUtils.preprocess import preprocess, overlaySegmentation

class SegNode(Node):
    def __init__(self):
        super.__init__('segNodes')
        self.declareParameter(
            'enginePat'
            'singleSeg.engine'
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
        self.pub = self.create_publisher(
            Image,
            '/seg/image',
            10
        )

    def imageCb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        inp, vis = preprocess(frame)
        outputs = self.trtModel.infer(inp)
        seg = outputs['segmentation']
        vis = overlaySegmentation(
            vis,
            seg
        )

        outMsg = self.bridge.cv2_to_imgmsg(
            vis,
            encoding = 'bgr8'
        )
        outMsg.header = Header(
            stamp = self.get_clock().now().to_msg(),
            frame_id = msg.header.frame_id
        ) 
        self.pub.publish(outMsg)

def main(args = None):
    rclpy.init(args=args)
    node = SegNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()