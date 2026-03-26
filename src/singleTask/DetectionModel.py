from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.detectionHead import DetectionHead
import torch
import torch.nn as nn

class SingleTaskDetectionModel(nn.Module):
    def __init__(self, numClasses: int):
        super().__init__()
        self.backbone = MobileNetV3Lite()
        self.head = DetectionHead(inChannels=576, numClasses=numClasses)

    def forward(self, x):
        feats = self.backbone(x)
        preds = self.head(feats)
        return preds