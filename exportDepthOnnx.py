import torch
from src.singleTask.singleDepthModel import SingleDepthModel

def main():
    model = SingleDepthModel(
        imgSize = (480, 640)    
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
        "singleDepth.onnx",
        input_names=["input"],
        output_names=["depth"],
        opset_version=12,
        dynamic_axes={
            "input":{0: "batch"},
            "depth":{0: "batch"}
        }
    )

    print("Exported to singleDepth.onnx")

if __name__ == "__main__":
    main()