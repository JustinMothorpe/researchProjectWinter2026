from .mobileNetv3 import MobileNetV3Lite

def buildBackbone(name: str):
    if name == "mobilenetV3":
        return MobileNetV3Lite()
    else:
        raise ValueError(f"Backbone '{name}' not supported.")