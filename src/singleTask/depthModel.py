import torch.nn as nn
from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.depthHead import DepthHead

class SingleTaskSegmentationModel(nn.Module):
    def __init__(
            self, 
            imgSize: int
    ):
        super().__init__()
        
        if imgSize is None:
            imgSize = (480, 640)
        
        self.backbone = MobileNetV3Lite()
        self.head = DepthHead(
            inChannels = 578,
            outSize = imgSize
        )
    
    def forward(self, x):
        feats = self.backbone(x)
        depth = self.head(feats)
        return depth