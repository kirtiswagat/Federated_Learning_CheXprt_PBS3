"""CheXpert loading: U-Zero labels, frontal filter, 320x320 preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import LABELS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def u_zero(frame: pd.DataFrame, labels: tuple[str, ...] = LABELS) -> pd.DataFrame:
    """Map uncertain (-1) and unmentioned (NaN) labels to 0."""
    result = frame.copy()
    values = result[list(labels)].apply(pd.to_numeric, errors="coerce")
    result[list(labels)] = values.eq(1).astype("float32")
    return result


def prepare_frame(
    frame: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    frontal_only: bool = True,
    uncertain_policy: str = "zero",
) -> pd.DataFrame:
    if uncertain_policy != "zero":
        raise NotImplementedError("Only the U-Zero policy is implemented")
    result = frame
    if frontal_only and "Frontal/Lateral" in result.columns:
        result = result[result["Frontal/Lateral"] == "Frontal"]
    return u_zero(result, labels).reset_index(drop=True)


def class_weights(
    frame: pd.DataFrame, labels: tuple[str, ...] = LABELS, cap: float = 10.0
) -> torch.Tensor:
    """BCE pos_weight = negatives / positives, capped for rare labels."""
    positives = frame[list(labels)].to_numpy(dtype=np.float64).sum(axis=0)
    negatives = len(frame) - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)
    return torch.tensor(np.clip(weights, 1.0, cap), dtype=torch.float32)


class CheXpertDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        labels: tuple[str, ...] = LABELS,
        image_size: tuple[int, int] = (320, 320),
        trigger: dict | None = None,
        trigger_rows: set[int] | None = None,
    ):
        """trigger = {size, location, intensity}; applied to trigger_rows (all rows if None)."""
        self.frame = frame.reset_index(drop=True)
        self.paths = self.frame["Path"].tolist()
        self.targets = self.frame[list(labels)].to_numpy(dtype=np.float32)
        self.image_root = Path(image_root)
        self.size = (image_size[1], image_size[0])
        self.trigger = trigger
        self.trigger_rows = trigger_rows

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.image_root / self.paths[index]) as image:
            image = image.convert("RGB").resize(self.size, Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        if self.trigger and (self.trigger_rows is None or index in self.trigger_rows):
            row, column = self.trigger["location"]
            side = int(self.trigger["size"])
            pixels = pixels.copy()
            pixels[row : row + side, column : column + side, :] = self.trigger[
                "intensity"
            ]
        array = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(array.transpose(2, 0, 1).copy()),
            torch.from_numpy(self.targets[index]),
        )
