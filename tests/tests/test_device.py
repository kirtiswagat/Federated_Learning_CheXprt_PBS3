import torch

from src.device import resolve_device


def test_resolve_device_returns_supported_type():
    assert resolve_device().type in {"cuda", "mps", "cpu"}


def test_cuda_requirement_is_explicit():
    if not torch.cuda.is_available():
        import pytest

        with pytest.raises(RuntimeError, match="CUDA is required"):
            resolve_device(require_cuda=True)
