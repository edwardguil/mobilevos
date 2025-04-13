# mobilevos/modules/backbone.py
import torch.nn as nn
import torchvision.models as models

def get_backbone(name):
    if name.lower() == "resnet18":
        model = models.resnet18(pretrained=True)
        # Remove fully connected layers; get feature maps at a reduced resolution.
        layers = list(model.children())[:-2]
        backbone = nn.Sequential(*layers)
        backbone.out_channels = 512
        return backbone
    elif name.lower() == "resnet50":
        model = models.resnet50(pretrained=True)
        layers = list(model.children())[:-2]
        backbone = nn.Sequential(*layers)
        backbone.out_channels = 2048
        return backbone
    elif name.lower() == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
        # Use features from mobilenet
        backbone = model.features
        backbone.out_channels = 1280
        return backbone
    else:
        raise ValueError(f"Unknown backbone name: {name}")