"""Run one CBRA sensitivity experiment (R12-R19): ResNet-50, MPS, frozen inputs."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
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
from src.monitoring.artifact_registry import write_manifest
from src.p3.artifacts import validate_p3_run
from src.p3.cbra import cbra_aggregate, compute_contributions
from src.p3.registry import P3RunRegistry
from src.reproducibility import load_checkpoint_cpu, seed_everything

P3_ALIASES = (
    ("endpoint_per_label_metrics.csv", "per_label_metrics.csv"),
    ("endpoint_macro_metrics.csv", "macro_metrics.csv"),
    ("endpoint_valid_predictions.csv", "best_valid_predictions.csv"),
    ("endpoint_model.pt", "checkpoint.pt"),
    ("round_history.csv", "training_summary.csv"),
    ("aggregation_details.csv", "cbra_round_details.csv"),
    ("aggregation_details.csv", "cbra_weights.csv"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="R12 ... R19")
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
        help="Mid-client checkpoint every N batches (0 disables)",
    )
    parser.add_argument("--rounds", type=int, help="Override rounds (smoke tests only)")
    parser.add_argument("--max-train-per-client", type=int, help="Smoke tests only")
    parser.add_argument("--max-valid", type=int, help="Smoke tests only")
    parser.add_argument("--tag", help="Suffix for the output folder, e.g. smoke")
    args = parser.parse_args()

    root = args.root.resolve()
    base = root / "Output" / "p1_critical_revision"
    spec = P3RunRegistry(root).get(args.run_id)
    config = RunRegistry(root).get("R03")  # FedAvg, Dirichlet 0.1, 5 clients
    if (
        config.federated.dirichlet_alpha != spec.dirichlet_alpha
        or config.federated.clients != spec.clients
    ):
        raise RuntimeError("P3 spec does not match the frozen R03 partition settings")
    rounds = args.rounds or spec.rounds
    config = replace(
        config,
        paths=ProjectPaths(root, output_dir=root / "Output" / "p3_cbra"),
        federated=replace(config.federated, rounds=rounds),
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
    client_labels = [loader.dataset.targets for loader in loaders]
    global_labels = np.concatenate(client_labels)
    print(
        f"  {args.run_id} ({spec.variant}): alpha_n={spec.alpha_n} beta_q={spec.beta_q} "
        f"gamma_d={spec.gamma_d} rounds={rounds} clients={[len(l.dataset) for l in loaders]} "
        f"valid={len(valid_frame)} workers={workers} device={device.type}",
        flush=True,
    )

    def aggregate(states, sizes, global_state, round_id):
        table = compute_contributions(
            states,
            global_state,
            sizes,
            client_labels,
            global_labels,
            spec.alpha_n,
            spec.beta_q,
            spec.gamma_d,
        )
        table.insert(0, "round", round_id)
        print(
            f"[{run_name}] round {round_id} CBRA weights "
            f"{[round(w, 4) for w in table['normalized_weight']]}",
            flush=True,
        )
        return cbra_aggregate(states, table), table.to_dict("records")

    print("[3/4] Initialising model", flush=True)
    runner = FederatedRunner(
        config,
        lambda: build_resnet50(config.model.num_labels, False, config.model.dropout),
        device,
        initial_state=load_checkpoint_cpu(initial_checkpoint),
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        aggregate_fn=aggregate,
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
    for source, alias in P3_ALIASES:
        shutil.copy2(output / source, output / alias)
    final = pd.read_csv(output / "round_history.csv").iloc[-1].to_dict()
    (output / f"{args.run_id.lower()}_results.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "variant": spec.variant,
                "alpha_n": spec.alpha_n,
                "beta_q": spec.beta_q,
                "gamma_d": spec.gamma_d,
                "final_metrics": final,
            },
            indent=2,
        )
        + "\n"
    )
    write_manifest(output, run_name, {"device_type": device.type, "source_stage": "P3"})
    validate_p3_run(output)
    print(f"Done: {output}", flush=True)
    print(f"Checksums: {output / 'artifact_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
