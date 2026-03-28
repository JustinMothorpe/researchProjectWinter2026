import torch.nn as nn
from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.depthHead import DepthHead

class SingleDepthModel(nn.Module):
    def __init__(
            self, 
            imgSize: int
    ):
        super().__init__()
        
        if imgSize is None:
            imgSize = (480, 640)
        self.backBone = MobileNetV3Lite()
        self.depthHead = DepthHead(
            576,
            imgSize
        ) 

    def forward(self, x):
        feats = self.backBone(x)
        return self.depthHead(feats)