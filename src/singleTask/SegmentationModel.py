import torch.nn as nn
from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.segmentationHead import SegmentationHead

class SingleTaskSegmentationModel(nn.Module):
    def __init__(
            self, 
            numClasses: int,
            imgSize = (480, 640)
    ):
        super().__init__()
        self.backbone = MobileNetV3Lite()
        self.head = SegmentationHead(
            inChannels = 578,
            numClasses = numClasses,
            outSize = imgSize
        )
    
    def forward(self, x):
        feats = self.backbone(x)
        maskLogits = self.head(feats)
        return maskLogits