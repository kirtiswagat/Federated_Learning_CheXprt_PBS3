"""Run one prespecified P1 federated experiment (ResNet-50, MPS)."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from src.config import ProjectPaths
from src.device import resolve_mps
from src.federated.p1_adapter import (
    build_data,
    make_validation_fn,
    partition_name,
    verify_checksum,
)
from src.federated.run_registry import RunRegistry
from src.federated.runner import FederatedRunner
from src.models.resnet_multilabel import build_resnet50
from src.reproducibility import load_checkpoint_cpu, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder containing CheXpert-v1.0-small/",
    )
    parser.add_argument(
        "--valid-csv",
        type=Path,
        help="Default: <data-root>/CheXpert-v1.0-small/val_split.csv",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save a mid-client checkpoint every N batches (0 disables)",
    )
    parser.add_argument(
        "--rounds", type=int, help="Override frozen rounds (smoke tests only)"
    )
    parser.add_argument("--max-train-per-client", type=int, help="Smoke tests only")
    parser.add_argument("--max-valid", type=int, help="Smoke tests only")
    parser.add_argument("--tag", help="Suffix for the output folder, e.g. smoke")
    args = parser.parse_args()

    root = args.root.resolve()
    base = root / "Output" / "p1_critical_revision"
    config = RunRegistry(root).get(args.run_id)
    config = replace(config, paths=ProjectPaths(root, output_dir=base / "runs"))
    if args.rounds:
        config = replace(
            config, federated=replace(config.federated, rounds=args.rounds)
        )
    workers = (
        config.runtime.num_workers if args.num_workers is None else args.num_workers
    )
    device = resolve_mps(config.runtime.require_mps)
    seed_everything(config.training.seed)

    partition_csv = (
        base / "partitions" / f"{partition_name(config)}_client_assignment.csv"
    )
    initial_checkpoint = base / "frozen" / "initial_checkpoint.pt"
    print("[1/4] Verifying frozen inputs", flush=True)
    verify_checksum(partition_csv, base / "partitions.sha256")
    verify_checksum(initial_checkpoint, base / "frozen" / "frozen.sha256")

    print("[2/4] Building data loaders", flush=True)
    valid_csv = (
        args.valid_csv or args.data_root / "CheXpert-v1.0-small" / "val_split.csv"
    )
    run_name = f"{args.run_id}_{args.tag}" if args.tag else args.run_id
    run_dir = config.paths.output_dir / run_name
    loaders, valid_loader, valid_frame, weights = build_data(
        config,
        partition_csv,
        valid_csv,
        args.data_root,
        workers,
        args.max_train_per_client,
        args.max_valid,
    )
    print(
        f"  {args.run_id}: {config.federated.method} {partition_name(config)} "
        f"rounds={config.federated.rounds} clients={[len(l.dataset) for l in loaders]} "
        f"valid={len(valid_frame)} workers={workers} device={device.type}",
        flush=True,
    )

    print("[3/4] Initialising model", flush=True)
    factory = lambda: build_resnet50(
        config.model.num_labels, False, config.model.dropout
    )
    runner = FederatedRunner(
        config,
        factory,
        device,
        initial_state=load_checkpoint_cpu(initial_checkpoint),
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))

    print("[4/4] Training", flush=True)
    output = runner.run(
        loaders,
        make_validation_fn(valid_loader, valid_frame, config, device, run_dir),
        criterion,
        run_name,
        resume=args.resume,
        eval_batches=len(valid_loader),
    )
    print(f"Done: {output}", flush=True)
    print(f"Checksums: {output / 'artifact_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
