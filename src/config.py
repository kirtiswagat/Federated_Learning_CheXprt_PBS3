##########----------------------------------------------------------------------#######
#########----------src/config-------------------------------------------------#######
"""Frozen, serializable configuration for the P1 experiments."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

LABELS = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    train_csv: Path | None = None
    valid_csv: Path | None = None
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "train_csv", self.train_csv or root / "train.csv")
        object.__setattr__(self, "valid_csv", self.valid_csv or root / "valid.csv")
        object.__setattr__(
            self,
            "output_dir",
            self.output_dir or root / "outputs" / "p1_critical_revision",
        )


@dataclass(frozen=True)
class DatasetConfig:
    labels: tuple[str, ...] = LABELS
    uncertain_policy: str = "zero"
    frontal_only: bool = True


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    num_labels: int = 14
    dropout: float = 0.3
    image_size: tuple[int, int] = (320, 320)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    precision: str = "fp32"
    local_epochs: int = 1
    threshold: float = 0.5


@dataclass(frozen=True)
class FederatedConfig:
    clients: int = 5
    rounds: int = 5
    partition: str = "iid"
    dirichlet_alpha: float | None = None
    method: str = "FedAvg"
    fedprox_mu: float = 0.001


@dataclass(frozen=True)
class BlockchainConfig:
    validators: tuple[str, ...] = ("validator_1", "validator_2", "validator_3")
    majority: int = 2
    secure: bool = False  # signed votes, verified client signatures, round commitments


@dataclass(frozen=True)
class StatisticalConfig:
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    correction: str = "benjamini-hochberg"


@dataclass(frozen=True)
class RuntimeConfig:
    prefer_cuda: bool = False
    require_cuda: bool = False
    allow_mps: bool = True
    device: str = "mps"
    require_mps: bool = True
    num_workers: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    paths: ProjectPaths
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    statistics: StatisticalConfig = field(default_factory=StatisticalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if self.model.num_labels != len(self.dataset.labels):
            raise ValueError("model.num_labels must match the frozen label order")
        if self.dataset.uncertain_policy not in {"zero", "one", "ignore", "mapping"}:
            raise ValueError("uncertain_policy must be zero, one, ignore, or mapping")
        if (
            self.federated.partition == "dirichlet"
            and self.federated.dirichlet_alpha is None
        ):
            raise ValueError("Dirichlet runs require dirichlet_alpha")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["paths"] = {key: str(item) for key, item in value["paths"].items()}
        return value

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def default_config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(paths=ProjectPaths(root))


def environment_report() -> dict[str, Any]:
    try:
        import torch

        cuda = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_available else None
        mps = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        torch_version = torch.__version__
    except ImportError:
        cuda = gpu = torch_version = None
        cuda_available = mps = False
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch_version,
        "cuda": cuda,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "mps_available": mps,
    }


def run_configs(root: Path) -> Mapping[str, ExperimentConfig]:
    base = default_config(root)
    values = {
        "R01": FederatedConfig(clients=5, rounds=10, partition="iid", method="FedAvg"),
        "R02": FederatedConfig(
            clients=5,
            rounds=5,
            partition="dirichlet",
            dirichlet_alpha=0.3,
            method="FedAvg",
        ),
        "R03": FederatedConfig(
            clients=5,
            rounds=5,
            partition="dirichlet",
            dirichlet_alpha=0.1,
            method="FedAvg",
        ),
        "R04": FederatedConfig(
            clients=5, rounds=5, partition="iid", method="FedProx", fedprox_mu=0.001
        ),
        "R05": FederatedConfig(
            clients=5,
            rounds=5,
            partition="dirichlet",
            dirichlet_alpha=0.1,
            method="FedProx",
            fedprox_mu=0.001,
        ),
    }
    return MappingProxyType(
        {run_id: replace(base, federated=config) for run_id, config in values.items()}
    )
    