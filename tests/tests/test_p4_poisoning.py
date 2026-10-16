import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import LABELS
from src.data.chexpert_dataset import CheXpertDataset
from src.robustness.config import P4Config, RobustnessPaths
from src.robustness.harness import ProvidedUpdateAttack, RobustnessHarness
from src.robustness.poisoning import (
    backdoor_frame,
    flip_all_labels,
    triggered_evaluation_rows,
)


def _frame(tmp_path, rows=10):
    for index in range(rows):
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(
            tmp_path / f"{index}.png"
        )
    data = {"Path": [f"{i}.png" for i in range(rows)]}
    for position, label in enumerate(LABELS):
        data[label] = [float((i + position) % 2) for i in range(rows)]
    return pd.DataFrame(data)


def test_flip_all_labels_inverts_every_target(tmp_path):
    frame = _frame(tmp_path)
    flipped = flip_all_labels(frame)
    assert (
        flipped[list(LABELS)].to_numpy() == 1 - frame[list(LABELS)].to_numpy()
    ).all()
    assert (
        frame[list(LABELS)].to_numpy() == _frame(tmp_path)[list(LABELS)].to_numpy()
    ).all()


def test_backdoor_frame_is_deterministic_and_sets_target(tmp_path):
    frame = _frame(tmp_path)
    first, rows_a = backdoor_frame(frame, 0, 0.3, seed=7)
    second, rows_b = backdoor_frame(frame, 0, 0.3, seed=7)
    assert rows_a == rows_b and len(rows_a) == 3
    assert (first.loc[sorted(rows_a), LABELS[0]] == 1.0).all()
    untouched = sorted(set(range(len(frame))) - rows_a)
    assert (first.loc[untouched, LABELS[0]] == frame.loc[untouched, LABELS[0]]).all()


def test_trigger_only_on_selected_rows(tmp_path):
    frame = _frame(tmp_path)
    trigger = {"size": 2, "location": (0, 0), "intensity": 1.0}
    dataset = CheXpertDataset(frame, tmp_path, LABELS, (16, 16), trigger, {1})
    plain, _ = dataset[0]
    marked, _ = dataset[1]
    assert not torch.equal(plain[:, 0, 0], marked[:, 0, 0])
    assert torch.equal(plain[:, 5:, 5:], marked[:, 5:, 5:])
    assert marked[0, 0, 0] > plain[0, 0, 0]


def test_triggered_rows_exclude_target_positives(tmp_path):
    frame = _frame(tmp_path)
    rows = triggered_evaluation_rows(frame, 0, 100, seed=1)
    assert (rows[LABELS[0]] == 0).all() and len(rows) > 0


def test_provided_update_attack_submits_retrained_update(tmp_path):
    clean = {"w": torch.zeros(3)}
    poisoned = {"w": torch.ones(3)}
    result = ProvidedUpdateAttack("label_flipping", {"client_0": poisoned}).apply(
        "client_0", clean, None, {}, None, None
    )
    assert torch.equal(result.submitted_update["w"], poisoned["w"])
    assert result.malicious and result.attacked_digest != result.original_digest


def test_harness_filters_attacks(tmp_path):
    config = P4Config(RobustnessPaths(tmp_path, output_root=tmp_path / "out"))
    harness = RobustnessHarness(config, {}, ("label_flipping",))
    assert harness.attack_names == ("label_flipping",)
