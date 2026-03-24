from backbone.mobileNetv3 import MobileNetV3Lite
import torch

model = MobileNetV3Lite()
x = torch.randn(1, 3, 640, 480)
y = model(x)
print(y.shape)