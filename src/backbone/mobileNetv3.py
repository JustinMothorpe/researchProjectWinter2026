import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small
class MobileNetV3Lite(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()
        base = mobilenet_v3_small(weights="DEFAULT")

        # extracting only the feature layers (no classifier)
        self.features = base.features
        
        if output_stride == 8:
            self.convertToOs8()

    def convertToOs8(self):
        # modify the last lares to increase spatial resolution
        for layer in self.features[-3:]:
            if hasattr(layer, "conv"):
                layer.conv[0].stride = (1, 1)

    def forward(self, x):
        return self.features(x) #feature map