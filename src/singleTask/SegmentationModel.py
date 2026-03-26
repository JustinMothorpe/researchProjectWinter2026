import torch.nn as nn
from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.segmentationHead import SegmentationHead
    
class SingleTaskSegmentationModel(nn.Module):
    def __init__(
            self, 
            numClasses: int,
            imgSize: int
    ):
        super().__init__()
        if imgSize is None:
            imgSize = (480, 640)
        
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