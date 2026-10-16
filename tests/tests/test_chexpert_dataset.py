import numpy as np
import pandas as pd
from PIL import Image

from src.config import LABELS
from src.data.chexpert_dataset import CheXpertDataset, class_weights, prepare_frame


def _frame(path):
    row = {label: np.nan for label in LABELS}
    row.update(
        {"Path": path, "Frontal/Lateral": "Frontal", "Cardiomegaly": -1.0, "Edema": 1.0}
    )
    lateral = {**row, "Frontal/Lateral": "Lateral"}
    return pd.DataFrame([row, lateral])


def test_u_zero_and_frontal_filter(tmp_path):
    frame = prepare_frame(_frame("a.jpg"))
    assert len(frame) == 1
    assert frame["Cardiomegaly"].iloc[0] == 0.0 and frame["Edema"].iloc[0] == 1.0


def test_dataset_shape_and_class_weight_cap(tmp_path):
    Image.new("L", (50, 40)).save(tmp_path / "a.jpg")
    frame = prepare_frame(_frame("a.jpg"))
    image, target = CheXpertDataset(frame, tmp_path)[0]
    assert image.shape == (3, 320, 320) and target.shape == (14,)
    assert class_weights(frame, cap=10.0).max() <= 10.0
