"""The frozen ResNet-50/ImageNet multi-label model definition."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# IMAGENET1K_V2 weights, used when the download fails.
LOCAL_WEIGHTS = Path(__file__).resolve().parents[2] / "resnet50-11ad3fa6.pth"


def _imagenet_resnet50() -> nn.Module:
    try:
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception as error:
        if not LOCAL_WEIGHTS.is_file():
            raise RuntimeError(
                f"Download failed ({error}) and {LOCAL_WEIGHTS} is missing"
            ) from error
        print(f"Download failed; loading local weights {LOCAL_WEIGHTS}", flush=True)
        model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(LOCAL_WEIGHTS, map_location="cpu"))
        return model


def build_resnet50(
    num_labels: int = 14, pretrained: bool = True, dropout: float = 0.3
) -> nn.Module:
    model = _imagenet_resnet50() if pretrained else models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_labels)
    )
    return model
