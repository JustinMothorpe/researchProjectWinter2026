import torch
import torch.nn as nn

from src.backbone.mobileNetv3 import MobileNetV3Lite
from src.heads.segmentationHead import SegmentationHead
from src.heads.detectionHead import DetectionHead
from src.heads.depthHead import DepthHead

class MultiTaskModel(nn.Module):
    def __init__(
            self,
            numClassesDet: int,
            numClassesSeg: int,
            imgSize: tuple[int, int],
            numAnchors: int
    ):
        super().__init__()
        # Null variable Preventions:
        if imgSize is None:
            imgSize = (480, 640)
        
        if numAnchors is None:
            numAnchors = 1
        
        # Shared Backbone
        self.backBone = MobileNetV3Lite()

        # Feature map channels from MobileNetV3Lite
        inChannels = 576

        # Task Heads
        self.detHead = DetectionHead(
            inChannels=inChannels,
            numClasses=numClassesDet,
            numAnchors=numAnchors
        )

        self.segHead = SegmentationHead(
            inChannels=inChannels,
            numClasses=numClassesSeg,
            outSize=imgSize
        )

        self.depthHead = DepthHead(
            inChannels=inChannels,
            outSize=imgSize
        )

    def forward(self, x):
        #feature extraction
        feats = self.backBone(x)

        #Task-specific Predictions
        detOut = self.detHead(feats)
        segOut = self.segHead(feats)
        depthOut = self.depthHead(feats)

        # return the results cleanly
        return {
            "detection": detOut,
            "segmentation": segOut,
            "depth": depthOut
        }