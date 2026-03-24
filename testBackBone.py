from src.backbone import mobileNetv3
import torch

model = mobileNetv3.MobileNetV3Lite()
x = torch.randn(1, 3, 640, 480)
y = model(x)
print(y.shape)