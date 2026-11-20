"""Network registry: the ResNet-50 teacher and the three student architectures."""
from resnet50 import ResNet50
from mobilenetv2 import MobileNetV2
from squeezenet import SqueezeNet
from efficientnet_lite import EfficientNetLite0

REGISTRY = {
    "resnet50": ResNet50,
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}

PRETTY = {"resnet50": "ResNet-50 (teacher)", "mobilenetv2": "MobileNetV2",
          "squeezenet": "SqueezeNet 1.1", "efficientnet_lite0": "EfficientNet-Lite0"}


def build_model(arch, num_classes=100):
    if arch not in REGISTRY:
        raise ValueError(f"unknown architecture '{arch}', choose from {list(REGISTRY)}")
    return REGISTRY[arch](num_classes)
