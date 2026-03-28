import torch.nn as nn
from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.detectionHead import DetectionHead

class SingleDetModel(nn.Module):
    def __init__(
        self, 
        numClassesDet = 1, 
        numAnchors = 1
    ):
        super().__init__()
        self.backBone = MobileNetV3Lite()
        self.detHead = DetectionHead(
            576,
            numClassesDet,
            numAnchors
        )

    def forward(self, x):
        feats = self.backBone(x)
        return self.detHead(feats)