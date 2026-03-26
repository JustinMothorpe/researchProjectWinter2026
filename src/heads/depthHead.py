import torch
import torch.nn as nn
import torch.nn.functional as functional

class DepthHead(nn.Module):
    def __init__(self, inChannels: int, outSize: tuple[int, int]):
        super().__init__()
        self.outH, self.outW = outSize

        self.decoder = nn.Sequential(
            nn.Conv2d(
                inChannels,
                256,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                256,
                128,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                128,
                1,
                kernel_size = 1
            )
        )
    def forward(self, x):
        x = self.decoder(x)    #   [B, 1, H, W]
        x = functional.interpolate(
            x,
            size = (self.outH, self.outW),
            mode = "bilinear",
            align_corners = False
        )
        #   later I will constrian the range of it
        return x # [B, 1, outh, outw]