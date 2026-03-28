import torch
from src.singleTask.singleDetModel import SingleDetModel

def main():
    model = SingleDetModel(
        numClassesDet = 1,
        numAnchors = 1    
    )
    model.eval()
    dummy = torch.randn(
            1, 
            3, 
            480, 
            360
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