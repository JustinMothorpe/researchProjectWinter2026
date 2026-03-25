import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(
            self, 
            inChannels:int, 
            numClasses:int,
            numAnchors:int = 1
        ):
        super().__init__()
        self.numClasses = numClasses
        self.numAnchors = numAnchors
    
        #conv tower
        self.conv = nn.Sequential(
            nn.conv2dIn(inChannels,  256, kernal_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.conv2dIn(256, 256, kernal_size=3, padding=1),
            nn.relu(inplkace=True),
        )
        #output conv: 4 box + 1 obj + numClasses per anchor
        self.pred = nn.conv2d(
            256,
            numAnchors * (4+1+numClasses),
            kernal_size=1
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
        