import torch
import torch.nn as nn
"""class SingleTaskDetectionModel(nn.Module):
    def __init__(
            self, 
            inChannels: int,
            numClasses: int,
            numAnchors: int
        ):
        super().__init__()
        if numAnchors is None:
            numAnchors = 1
        self.numClasses = int(numClasses)
        self.numAnchors = int(numAnchors)
        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels,
                256,
                kernel_size=3,
                padding=1                
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        outChannels = self.numAnchors * (4 + 1 + self.numClasses)
        self.pred = nn.Conv2d"""
class DetectionHead(nn.Module):
    def __init__(
            self, 
            inChannels: int,
            numClasses: int,
            numAnchors: int
        ):
        super().__init__()
        if numAnchors is None:
            numAnchors = 1
        self.numClasses = int(numClasses)
        self.numAnchors = int(numAnchors)
        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels,
                256,
                kernel_size=3,
                padding=1                
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        outChannels = self.numAnchors * (4 + 1 + self.numClasses)
        self.pred = nn.Conv2d(
            256,
            outChannels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.conv(x) # [B, 256, H, W]
        x = self.pred(x) # [B, numAnchors * (4+1+numClasses), H, W]

        B, C, H, W = x.shape
        x = x.view(
            B, 
            self.numAnchors, 
            4 + 1 + self.numClasses, 
            H, 
            W
        )
        #[B, numAnchors, 4+1+numClasses, H, W]
        return x
        