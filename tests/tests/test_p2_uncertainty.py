import numpy as np

from src.p2.uncertainty import transform_uncertainty


def test_ignore_policy_masks_uncertain_labels():
    values, mask = transform_uncertainty(
        np.array([[1, -1], [0, 1]], dtype=float), "ignore"
    )
    assert values[0, 1] == 0 and mask[0, 1] == 0


def test_mapping_policy_is_label_specific():
    values, mask = transform_uncertainty(
        np.array([[-1, -1]], dtype=float), "mapping", {0: 1, 1: 0}
    )
    assert values.tolist() == [[1.0, 0.0]] and mask.all()
