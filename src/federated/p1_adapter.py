"""CheXpert/ResNet-50 adapter wiring data, checksums and evaluation to the runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import LABELS, ExperimentConfig
from src.data.chexpert_dataset import CheXpertDataset, class_weights, prepare_frame
from src.evaluation.metrics import evaluate_predictions


def partition_name(config: ExperimentConfig) -> str:
    fed = config.federated
    suffix = f"_{fed.dirichlet_alpha}" if fed.dirichlet_alpha else ""
    return f"{fed.partition}_{fed.clients}{suffix}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, checksum_file: Path) -> None:
    """Match by file name so checksum files stay valid after copying to Mac B."""
    expected = {}
    for line in checksum_file.read_text().splitlines():
        if line.strip():
            value, name = line.split(maxsplit=1)
            expected[Path(name.strip().lstrip("*")).name] = value
    if path.name not in expected:
        raise RuntimeError(f"{path.name} is not listed in {checksum_file}")
    if _sha256(path) != expected[path.name]:
        raise RuntimeError(f"Checksum mismatch for {path} (see {checksum_file})")


def _loader(
    frame: pd.DataFrame,
    image_root: Path,
    config: ExperimentConfig,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    dataset = CheXpertDataset(
        frame, image_root, config.dataset.labels, config.model.image_size
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator if shuffle else None,
        persistent_workers=workers > 0,
    )


def build_data(
    config: ExperimentConfig,
    partition_csv: Path,
    valid_csv: Path,
    image_root: Path,
    workers: int,
    max_train_per_client: int | None = None,
    max_valid: int | None = None,
):
    """Returns client loaders, validation loader/frame and BCE class weights."""
    labels, seed = config.dataset.labels, config.training.seed
    frontal = config.dataset.frontal_only
    assigned = prepare_frame(pd.read_csv(partition_csv), labels, frontal)
    weights = class_weights(assigned, labels)
    loaders = []
    for client in range(config.federated.clients):
        frame = assigned[assigned["client_id"] == client]
        if len(frame) == 0:
            raise RuntimeError(f"Client {client} has no images in {partition_csv}")
        if max_train_per_client:
            frame = frame.sample(
                min(max_train_per_client, len(frame)), random_state=seed
            )
        loaders.append(
            _loader(
                frame,
                image_root,
                config,
                config.training.batch_size,
                True,
                workers,
                seed + client,
            )
        )
    valid = prepare_frame(pd.read_csv(valid_csv), labels, frontal)
    if max_valid:
        valid = valid.sample(min(max_valid, len(valid)), random_state=seed)
    valid = valid.reset_index(drop=True)
    valid_loader = _loader(valid, image_root, config, 32, False, workers, seed)
    first = image_root / assigned["Path"].iloc[0]
    if not first.is_file():
        raise FileNotFoundError(f"Image not found: {first} (check --data-root)")
    return loaders, valid_loader, valid, weights


def make_validation_fn(
    valid_loader: DataLoader,
    valid_frame: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
) -> Callable:
    labels = config.dataset.labels
    targets = valid_frame[list(labels)].to_numpy(dtype=np.float32)
    patient = valid_frame["Path"].str.extract(r"(patient\d+)")[0]

    def validate(
        model: torch.nn.Module, round_id: int, on_batch: Callable[[int], None]
    ):
        model.eval()
        outputs = []
        with torch.no_grad():
            for images, _ in valid_loader:
                outputs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
                on_batch(len(images))
        probabilities = np.concatenate(outputs)
        per_label, macro = evaluate_predictions(
            targets, probabilities, labels, config.training.threshold
        )
        predictions = pd.concat(
            [
                pd.DataFrame({"Path": valid_frame["Path"], "patient_id": patient}),
                pd.DataFrame(targets, columns=[f"y_{label}" for label in labels]),
                pd.DataFrame(probabilities, columns=[f"p_{label}" for label in labels]),
            ],
            axis=1,
        )
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        predictions.to_csv(
            run_dir / "predictions" / f"round_{round_id:02d}_valid_predictions.csv",
            index=False,
        )
        predictions.to_csv(run_dir / "endpoint_valid_predictions.csv", index=False)
        per_label.to_csv(
            run_dir / "metrics" / f"round_{round_id:02d}_per_label.csv", index=False
        )
        macro.to_csv(
            run_dir / "metrics" / f"round_{round_id:02d}_macro.csv", index=False
        )
        return per_label, macro

    return validate
