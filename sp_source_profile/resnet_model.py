"""ResNet152 model initialization for morphology classification."""

from typing import Any
import torch.nn as nn
from torchvision import models


def initialize_resnet152(num_classes: int, fast_training: bool = False) -> nn.Module:
    """Create a ResNet152 model with LogSoftmax classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    fast_training : bool
        If True, freeze early layers (not needed for inference).

    Returns
    -------
    nn.Module
        ResNet152 with custom classifier head.
    """
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

    if fast_training:
        for param in model.parameters():
            param.requires_grad = False
        for layer in [model.layer3, model.layer4, model.fc]:
            for param in layer.parameters():
                param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.LogSoftmax(dim=1),
    )
    return model
