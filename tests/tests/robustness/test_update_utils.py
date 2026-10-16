import torch
import pytest

from src.robustness.update_utils import (
    state_dict_to_update,
    update_to_state_dict,
    flatten_update,
    unflatten_update,
    validate_update_schema,
)


def test_update_round_trip():
    global_state = {"w": torch.zeros(2)}
    local_state = {"w": torch.ones(2)}
    update = state_dict_to_update(local_state, global_state)
    flat, metadata = flatten_update(update)
    assert torch.equal(
        update_to_state_dict(global_state, unflatten_update(flat, metadata))["w"],
        local_state["w"],
    )


def test_schema_mismatch_rejected():
    with pytest.raises(ValueError):
        validate_update_schema({"bad": torch.ones(1)}, {"w": torch.ones(1)})
