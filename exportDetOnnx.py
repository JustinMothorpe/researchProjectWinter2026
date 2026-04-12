import torch
from src.singleTask.singleDetModel import SingleDetModel
from .src.sharedUtils.sizeMain import MainSize

def main():
    size = MainSize(480, 640)

    model = SingleDetModel(
        numClassesDet = 1,
        numAnchors = 1    
    )
    model.eval()
    dummy = torch.randn(
            1, 
            3, 
            size.getSize(False), 
            size.getSize(True)
        )
    
    torch.onnx.export(
        model,
        dummy,
        "singleDet.onnx",
        input_names=["input"],
        output_names=["detection"],
        opset_version=12,
        dynamic_axes={
            "input":{0: "batch"},
            "detection":{0: "batch"}
        }
    )

    print("Exported to singleDet.onnx")

if __name__ == "__main__":
    main()