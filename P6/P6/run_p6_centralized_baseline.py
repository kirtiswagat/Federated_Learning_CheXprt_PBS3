"""Centralized ResNet-50 reference on the pooled revised training set (matched budget)."""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ProjectPaths  # noqa: E402
from src.device import resolve_mps  # noqa: E402
from src.federated.p1_adapter import (  # noqa: E402
    build_data,
    make_validation_fn,
    verify_checksum,
)
from src.federated.run_registry import RunRegistry  # noqa: E402
from src.models.resnet_multilabel import build_resnet50  # noqa: E402
from src.reproducibility import load_checkpoint_cpu, seed_everything  # noqa: E402


def fmt(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train", type=int, help="Smoke test only")
    parser.add_argument("--max-valid", type=int, help="Smoke test only")
    parser.add_argument("--tag", default="")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--fresh", action="store_true", help="Ignore any checkpoint")
    args = parser.parse_args()

    base = ROOT / "Output" / "p1_critical_revision"
    config = RunRegistry(ROOT).get("R01")
    config = replace(config, paths=ProjectPaths(ROOT, output_dir=ROOT / "Output"))
    device = resolve_mps(config.runtime.require_mps)
    seed_everything(config.training.seed)

    partition_csv = base / "partitions" / "iid_5_client_assignment.csv"
    initial = base / "frozen" / "initial_checkpoint.pt"
    verify_checksum(partition_csv, base / "partitions.sha256")
    verify_checksum(initial, base / "frozen" / "frozen.sha256")

    valid_csv = args.data_root / "CheXpert-v1.0-small" / "val_split.csv"
    per_client = args.max_train // 5 if args.max_train else None
    loaders, valid_loader, valid_frame, weights = build_data(
        config,
        partition_csv,
        valid_csv,
        args.data_root,
        args.num_workers,
        per_client,
        args.max_valid,
    )
    pooled = ConcatDataset([loader.dataset for loader in loaders])
    batch_size = config.training.batch_size

    def epoch_loader(epoch: int, done_batches: int) -> DataLoader:
        # Per-epoch permutation keyed on (seed, epoch) so a resume sees the same order.
        order = torch.randperm(
            len(pooled),
            generator=torch.Generator().manual_seed(config.training.seed + epoch),
        )
        subset = Subset(pooled, order[done_batches * batch_size :].tolist())
        return DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
        )

    run_dir = ROOT / "Output" / "p6_centralized" / f"C01{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_resnet50(config.model.num_labels, False, config.model.dropout)
    model.load_state_dict(load_checkpoint_cpu(initial))
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    validate = make_validation_fn(valid_loader, valid_frame, config, device, run_dir)

    print(
        f"images={len(pooled)} valid={len(valid_frame)} epochs={args.epochs} "
        f"device={device.type}",
        flush=True,
    )
    history = []
    total_batches = -(-len(pooled) // batch_size)
    valid_batches = len(valid_loader)
    run_start, eval_seconds = time.time(), 0.0
    checkpoint_path = run_dir / "resume_checkpoint.pt"
    # position of the next unit of work: (epoch, batches already done in it)
    state = {"epoch": 1, "batches": 0, "loss_sum": 0.0, "train_seconds": 0.0}
    if checkpoint_path.exists() and not args.fresh:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, eval_seconds = saved["history"], saved["eval_seconds"]
        state = saved["state"]
        print(
            f"RESUMING at epoch {state['epoch']} batch {state['batches']}/"
            f"{total_batches}",
            flush=True,
        )

    def save_checkpoint() -> None:
        tmp = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "eval_seconds": eval_seconds,
                "state": state,
            },
            tmp,
        )
        tmp.replace(checkpoint_path)

    for epoch in range(state["epoch"], args.epochs + 1):
        model.train()
        start = time.time() - state["train_seconds"]
        session_start = time.time()
        loss_sum, batches = state["loss_sum"], state["batches"]
        session_batches = 0
        for images, targets in epoch_loader(epoch, batches):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
            session_batches += 1
            if batches % args.checkpoint_every == 0:
                state = {
                    "epoch": epoch,
                    "batches": batches,
                    "loss_sum": loss_sum,
                    "train_seconds": time.time() - start,
                }
                save_checkpoint()
            if batches % args.progress_every == 0 or batches == total_batches:
                rate = session_batches / (time.time() - session_start)
                epoch_eta = (total_batches - batches) / rate
                eval_est = eval_seconds / (epoch - 1) if epoch > 1 else 0.0
                epochs_left = args.epochs - epoch
                total_eta = (
                    epoch_eta
                    + eval_est
                    + epochs_left * (total_batches / rate + eval_est)
                )
                print(
                    f"[epoch {epoch}/{args.epochs}] "
                    f"{batches}/{total_batches} ({100 * batches / total_batches:5.1f}%) "
                    f"loss {loss_sum / batches:.4f} "
                    f"{rate * config.training.batch_size:.1f} img/s "
                    f"epoch ETA {fmt(epoch_eta)} | total ETA {fmt(total_eta)} "
                    f"| elapsed {fmt(time.time() - run_start)}",
                    flush=True,
                )
        train_seconds = time.time() - start
        eval_start = time.time()
        seen = [0]

        def on_eval_batch(n: int) -> None:
            seen[0] += 1
            if seen[0] % 50 == 0 or seen[0] == valid_batches:
                done = time.time() - eval_start
                print(
                    f"[epoch {epoch}/{args.epochs}] evaluating "
                    f"{seen[0]}/{valid_batches} ETA "
                    f"{fmt(done / seen[0] * (valid_batches - seen[0]))}",
                    flush=True,
                )

        _, macro = validate(model, epoch, on_eval_batch)
        eval_seconds += time.time() - eval_start
        history.append(
            {
                "epoch": epoch,
                "train_seconds": train_seconds,
                "mean_loss": loss_sum / max(batches, 1),
            }
        )
        print(f"epoch {epoch} done\n{macro.to_string(index=False)}", flush=True)
        torch.save(model.state_dict(), run_dir / "endpoint_model.pt")
        state = {
            "epoch": epoch + 1,
            "batches": 0,
            "loss_sum": 0.0,
            "train_seconds": 0.0,
        }
        save_checkpoint()

    (run_dir / "centralized_summary.json").write_text(
        json.dumps(
            {
                "images": len(pooled),
                "epochs": args.epochs,
                "seed": config.training.seed,
                "history": history,
                "endpoint": "last epoch, no best-epoch selection",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
