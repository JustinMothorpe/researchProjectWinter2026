import torch
from src.multiTask.multiTaskModel import MultiTaskModel
from .src.sharedUtils.sizeMain import MainSize

size = MainSize(480, 640)

model = MultiTaskModel(
    numClassesDet=1,
    numClassesSeg=1,
    imgSize=(size.getSize(False), 
            size.getSize(True)),
    numAnchors=1
)

model.eval()

dummy = torch.randn(1, 3, 480, 640)

torch.onnx.export(
    model,
    dummy,
    "multiTaskModel.onnx",
    input_names=["input"],
    output_names=["detection", "segmentation", "depth"],
    opset_version=12,
    dynamic_axes={
        "input":{0: "batch"},
        "detection":{0: "batch"},
        "segmentation":{0: "batch"},
        "depth":{0: "batch"}
    }
)

print("export complete")