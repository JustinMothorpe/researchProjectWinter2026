import torch.nn as nn
from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.segmentationHead import SegmentationHead

class SingleSegModel(nn.Module):
    def __init__(
            self, 
            imgSize: int,
            numClassesSeg: int
    ):
        super().__init__()
        
        if imgSize is None:
            imgSize = (480, 640)
        if numClassesSeg is None:
            numClassesSeg = 1
        self.backBone = MobileNetV3Lite()
        self.segHead = SegmentationHead(
            576,
            numClassesSeg,
            imgSize
        ) 

    def forward(self, x):
        feats = self.backBone(x)
        return self.segHead(feats)