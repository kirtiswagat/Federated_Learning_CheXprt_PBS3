import torch

from src.reproducibility import load_checkpoint_cpu, save_checkpoint


def test_checkpoint_loads_on_cpu(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint({"weight": torch.ones(2)}, path)
    loaded = load_checkpoint_cpu(path)
    assert loaded["weight"].device.type == "cpu"
