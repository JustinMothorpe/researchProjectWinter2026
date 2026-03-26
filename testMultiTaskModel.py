from src.multiTask.multiTaskModel import MultiTaskModel
import torch
import time

deltaE = time.time()

model = MultiTaskModel(
    numClassesDet = 1,
    numClassesSeg = 1,
    imgSize = (480, 640),
    numAnchors = 1
    )

x = torch.randn(1, 3, 480, 640)

out = model(x)

deltaF = time.time()
deltaG = (deltaF-deltaE)
print("start time: " + str(deltaE))
print("end time: " + str(deltaF))
print("time diff: " + str(deltaG))

print(out["detection"].shape)
print(out["segmentation"].shape)
print(out["depth"].shape)