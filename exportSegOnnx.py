import torch
from src.singleTask.singlesegModel import SingleSegModel

def main():
    model = SingleSegModel(
        numClassesSeg = 1,
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
        "singleSeg.onnx",
        input_names=["input"],
        output_names=["segmentation"],
        opset_version=12,
        dynamic_axes={
            "input":{0: "batch"},
            "segmentation":{0: "batch"}
        }
    )

    print("Exported to singleSeg.onnx")

if __name__ == "__main__":
    main()