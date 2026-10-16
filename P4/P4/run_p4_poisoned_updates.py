"""Train the malicious clients on poisoned data for the label-flipping and backdoor attacks."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from P4.run_p4_real_attack_matrix import _atomic_save, _same_state
from src.config import ProjectPaths, run_configs
from src.data.chexpert_dataset import CheXpertDataset
from src.device import resolve_mps
from src.federated.client import train_client
from src.federated.p1_adapter import build_data, partition_name, verify_checksum
from src.models.resnet_multilabel import build_resnet50
from src.monitoring.progress import Progress
from src.reproducibility import load_checkpoint_cpu, seed_everything
from src.robustness.config import P4Config, RobustnessPaths
from src.robustness.poisoning import backdoor_frame, flip_all_labels
from src.robustness.registry import select_malicious_clients
from src.robustness.update_utils import state_dict_to_update

ATTACKS = ("label_flipping", "targeted_backdoor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-round", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Default: <source-round>/poisoned"
    )
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-train-per-client", type=int, help="Smoke tests only")
    args = parser.parse_args()

    root = args.root.resolve()
    base = root / "Output" / "p1_critical_revision"
    output_dir = args.output_dir or args.source_round / "poisoned"
    config = run_configs(root)["R03"]
    config = dataclasses.replace(
        config, paths=ProjectPaths(root, output_dir=base / "p4_source")
    )
    device = resolve_mps(config.runtime.require_mps)
    seed_everything(config.training.seed)
    p4 = P4Config(RobustnessPaths(root, output_root=output_dir))

    partition_csv = (
        base / "partitions" / f"{partition_name(config)}_client_assignment.csv"
    )
    initial_checkpoint = base / "frozen" / "initial_checkpoint.pt"
    verify_checksum(partition_csv, base / "partitions.sha256")
    verify_checksum(initial_checkpoint, base / "frozen" / "frozen.sha256")

    global_state = load_checkpoint_cpu(args.source_round / "global_state.pt")
    if not _same_state(global_state, load_checkpoint_cpu(initial_checkpoint)):
        raise RuntimeError(
            "source_round global_state differs from the frozen checkpoint"
        )

    needed = sorted(
        {
            client
            for fraction in p4.threat.malicious_fractions
            for client in select_malicious_clients(p4, fraction)
        }
    )
    print(f"[1/3] Malicious clients needed across fractions: {needed}", flush=True)
    loaders, _, _, weights = build_data(
        config,
        partition_csv,
        args.data_root / "CheXpert-v1.0-small" / "val_split.csv",
        args.data_root,
        args.num_workers,
        args.max_train_per_client,
        32,
    )
    labels = config.dataset.labels
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    trigger = {
        "size": p4.backdoor.trigger_size,
        "location": p4.backdoor.trigger_location,
        "intensity": p4.backdoor.trigger_intensity,
    }

    jobs = [
        (attack, client)
        for attack in args.attacks
        for client in needed
        if not (output_dir / attack / f"{client}_update.pt").is_file()
    ]
    print(
        f"[2/3] Training {len(jobs)} poisoned updates ({len(needed) * len(args.attacks) - len(jobs)} already done)",
        flush=True,
    )
    total = sum(len(loaders[int(client.split("_")[1])]) for _, client in jobs)
    progress = Progress(max(total, 1), args.progress_every)
    for attack, client in jobs:
        index = int(client.split("_")[1])
        frame = loaders[index].dataset.frame
        if attack == "label_flipping":
            poisoned, rows, use_trigger = flip_all_labels(frame, labels), None, None
        else:
            poisoned, rows = backdoor_frame(
                frame,
                p4.backdoor.target_label,
                p4.backdoor.poisoning_fraction,
                config.training.seed + index,
                labels,
            )
            use_trigger = trigger
        dataset = CheXpertDataset(
            poisoned, args.data_root, labels, config.model.image_size, use_trigger, rows
        )
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(
                config.training.seed + 2000 + index
            ),
            persistent_workers=args.num_workers > 0,
        )
        model = build_resnet50(config.model.num_labels, False, config.model.dropout).to(
            device
        )
        model.load_state_dict(global_state)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        tag = f"P4 poison {attack} {client}"
        steps = len(loader)

        def on_batch(batch, batch_loss, images, tag=tag, steps=steps):
            progress.advance(1, images)
            progress.tick(tag, f"batch {batch}/{steps} loss {batch_loss:.4f}", batch)

        state, loss, batches = train_client(
            model,
            loader,
            criterion,
            optimizer,
            device,
            config.training.local_epochs,
            global_state,
            0.0,
            on_batch=on_batch,
        )
        (output_dir / attack).mkdir(parents=True, exist_ok=True)
        _atomic_save(
            state_dict_to_update(state, global_state),
            output_dir / attack / f"{client}_update.pt",
        )
        print(
            f"  {attack} {client}: samples={len(dataset):,} batches={batches} loss={loss:.4f}",
            flush=True,
        )

    (output_dir / "poisoning_settings.json").write_text(
        json.dumps(
            {
                "label_flipping": "all 14 binary targets inverted for every sample of the malicious client",
                "targeted_backdoor": {
                    "target_label": labels[p4.backdoor.target_label],
                    "trigger": {
                        k: list(v) if isinstance(v, tuple) else v
                        for k, v in trigger.items()
                    },
                    "poisoning_fraction": p4.backdoor.poisoning_fraction,
                },
                "malicious_clients": needed,
                "local_epochs": config.training.local_epochs,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[3/3] Poisoned updates in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
