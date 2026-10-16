"""Execute the real one-round P4 attack matrix on frozen CheXpert inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ProjectPaths, run_configs
from src.data.chexpert_dataset import CheXpertDataset, prepare_frame
from src.device import resolve_mps
from src.evaluation.metrics import evaluate_predictions
from src.federated.client import train_client
from src.federated.p1_adapter import build_data, partition_name, verify_checksum
from src.models.resnet_multilabel import build_resnet50
from src.models.serialization import cpu_state_dict
from src.monitoring.progress import Progress
from src.reproducibility import (
    load_checkpoint_cpu,
    move_state_to_device,
    save_checkpoint,
    seed_everything,
)
from src.robustness.config import P4Config, RobustnessPaths
from src.robustness.harness import RobustnessHarness
from src.robustness.poisoning import triggered_evaluation_rows
from src.robustness.registry import build_condition_registry, select_malicious_clients
from src.robustness.update_utils import (
    compute_update_digest,
    state_dict_to_update,
    update_to_state_dict,
)


def _atomic_save(state, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_checkpoint(state, temporary)
    temporary.replace(path)


def _same_state(left, right) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key].cpu(), right[key].cpu()) for key in left
    )


def _evaluate_factory(
    valid_loader,
    valid_frame,
    config,
    device,
    progress=None,
    asr_loader=None,
    asr_target=0,
    asr_threshold=0.5,
):
    labels = config.dataset.labels
    targets = valid_frame[list(labels)].to_numpy(dtype=float)

    def evaluate(state, _condition):
        model = build_resnet50(config.model.num_labels, False, config.model.dropout).to(
            device
        )
        model.load_state_dict(state)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for images, _ in valid_loader:
                probabilities.append(
                    torch.sigmoid(model(images.to(device))).cpu().numpy()
                )
                if progress:
                    progress.advance(1, len(images))
                    progress.tick("P4 eval", "validation", progress.done)
        probabilities = __import__("numpy").concatenate(probabilities)
        # A poisoned model can output NaN; score it as uninformative and report the share.
        nonfinite = float((~__import__("numpy").isfinite(probabilities)).mean())
        probabilities = __import__("numpy").nan_to_num(
            probabilities, nan=0.5, posinf=1.0, neginf=0.0
        )
        per_label, macro = evaluate_predictions(
            targets, probabilities, labels, config.training.threshold
        )
        macro["nonfinite_prediction_fraction"] = nonfinite
        if asr_loader is not None and _condition.attack_name in (
            "targeted_backdoor",
            "clean",
        ):
            hits, count = 0, 0
            with torch.no_grad():
                for images, _ in asr_loader:
                    target_probability = torch.sigmoid(
                        model(images.to(device))[:, asr_target]
                    ).cpu()
                    hits += int((target_probability >= asr_threshold).sum())
                    count += len(images)
            macro["backdoor_asr"] = hits / max(count, 1)
            macro["backdoor_asr_images"] = count
        predictions = pd.DataFrame(
            {
                "Path": valid_frame["Path"],
                "patient_id": valid_frame["Path"].str.extract(r"(patient\d+)")[0],
            }
        )
        return per_label, macro, predictions

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Mid-client checkpoint every N batches while training (0 disables)",
    )
    parser.add_argument(
        "--source-round",
        type=Path,
        help="Reuse clean updates exported by another run (skips training)",
    )
    parser.add_argument("--max-train-per-client", type=int, help="Smoke tests only")
    parser.add_argument("--max-valid", type=int, help="Smoke tests only")
    parser.add_argument(
        "--aggregators",
        nargs="+",
        choices=(
            "fedavg",
            "coordinate_median",
            "trimmed_mean",
            "norm_clipped_fedavg",
        ),
        default=None,
        help="Aggregators to execute; default runs all four.",
    )
    parser.add_argument(
        "--clipping-threshold",
        default="median",
        help="norm_clipped_fedavg threshold: 'median' of submitted update norms, or a number",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="Run only these attacks (default: all five)",
    )
    parser.add_argument(
        "--poisoned-dir",
        type=Path,
        help="Retrained poisoned updates (default: <source-round>/poisoned); used for "
        "label_flipping and targeted_backdoor",
    )
    parser.add_argument("--asr-images", type=int, default=3000)
    args = parser.parse_args()

    root = args.root.resolve()
    base = root / "Output" / "p1_critical_revision"
    output_root = args.output_root or root / "Output" / "p4_robustness"
    config = run_configs(root)["R03"]
    config = __import__("dataclasses").replace(
        config, paths=ProjectPaths(root, output_dir=base / "p4_source")
    )
    device = resolve_mps(config.runtime.require_mps)
    seed_everything(config.training.seed)

    partition_csv = (
        base / "partitions" / f"{partition_name(config)}_client_assignment.csv"
    )
    initial_checkpoint = base / "frozen" / "initial_checkpoint.pt"
    verify_checksum(partition_csv, base / "partitions.sha256")
    verify_checksum(initial_checkpoint, base / "frozen" / "frozen.sha256")

    print("[1/4] Loading one-round P4 data", flush=True)
    valid_csv = args.data_root / "CheXpert-v1.0-small" / "val_split.csv"
    loaders, valid_loader, valid_frame, weights = build_data(
        config,
        partition_csv,
        valid_csv,
        args.data_root,
        args.num_workers,
        args.max_train_per_client,
        args.max_valid,
    )
    global_model = build_resnet50(
        config.model.num_labels, False, config.model.dropout
    ).to(device)
    global_model.load_state_dict(load_checkpoint_cpu(initial_checkpoint))
    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))

    print("[2/4] Preparing five clean client updates", flush=True)
    source_dir = args.source_round or output_root / "source_round"
    source_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_root / "source_round_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    state_path = source_dir / "global_state.pt"
    if state_path.is_file():
        if not _same_state(load_checkpoint_cpu(state_path), global_state):
            raise RuntimeError(
                f"{state_path} differs from this machine's frozen initial checkpoint"
            )
    elif args.source_round:
        raise FileNotFoundError(state_path)
    else:
        _atomic_save(global_state, state_path)
    train_progress = Progress(
        sum(len(loader) for loader in loaders) * config.training.local_epochs,
        args.progress_every,
    )
    clean_updates, sample_counts, local_datasets = {}, {}, {}
    for client_id, loader in enumerate(loaders):
        client = f"client_{client_id}"
        update_path = source_dir / f"{client}_update.pt"
        steps = len(loader) * config.training.local_epochs
        sample_counts[client] = len(loader.dataset)
        local_datasets[client] = loader.dataset.targets.copy()
        if update_path.is_file():
            clean_updates[client] = load_checkpoint_cpu(update_path)
            train_progress.advance(steps)
            print(f"  {client}: restored saved update", flush=True)
            continue
        if args.source_round:
            raise FileNotFoundError(update_path)
        generator = getattr(loader, "generator", None)
        if generator is not None:
            generator.manual_seed(config.training.seed + 1000 + client_id)
        tag = f"P4 train client_{client_id} {client_id + 1}/{len(loaders)}"

        def on_batch(batch, batch_loss, images, tag=tag, steps=steps):
            train_progress.advance(1, images)
            train_progress.tick(
                tag, f"batch {batch}/{steps} loss {batch_loss:.4f}", batch
            )

        local_model = build_resnet50(
            config.model.num_labels, False, config.model.dropout
        ).to(device)
        local_model.load_state_dict(global_state)
        optimizer = torch.optim.AdamW(
            local_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        mid_path = partial_dir / f"{client}_mid.pt"
        start_batch, start_loss = 0, 0.0
        if mid_path.is_file():
            mid = load_checkpoint_cpu(mid_path)
            local_model.load_state_dict(mid["model_state"])
            optimizer.load_state_dict(mid["optimizer_state"])
            start_batch, start_loss = int(mid["batch"]), float(mid["loss_sum"])
            train_progress.advance(start_batch)
            print(f"  {client}: resuming at batch {start_batch}/{steps}", flush=True)

        def on_checkpoint(
            batch,
            loss_sum,
            local_model=local_model,
            optimizer=optimizer,
            mid_path=mid_path,
        ):
            _atomic_save(
                {
                    "batch": batch,
                    "loss_sum": loss_sum,
                    "model_state": cpu_state_dict(local_model),
                    "optimizer_state": move_state_to_device(
                        optimizer.state_dict(), torch.device("cpu")
                    ),
                },
                mid_path,
            )

        state, loss, batches = train_client(
            local_model,
            loader,
            criterion,
            optimizer,
            device,
            config.training.local_epochs,
            global_state,
            0.0,
            on_batch=on_batch,
            start_batch=start_batch,
            start_loss=start_loss,
            checkpoint_every=args.checkpoint_every,
            on_checkpoint=on_checkpoint,
        )
        clean_updates[client] = state_dict_to_update(state, global_state)
        _atomic_save(clean_updates[client], update_path)
        mid_path.unlink(missing_ok=True)
        print(
            f"  {client}: samples={len(loader.dataset):,} batches={batches} loss={loss:.4f}",
            flush=True,
        )

    pd.DataFrame(
        [
            {"client_id": client, "sample_count": count}
            for client, count in sample_counts.items()
        ]
    ).to_csv(source_dir / "sample_counts.csv", index=False)
    update_digests = {
        client: compute_update_digest(update)
        for client, update in clean_updates.items()
    }
    (source_dir / "source_config.json").write_text(
        json.dumps(
            {
                "source_run": "R03 initial checkpoint, one local epoch",
                "clients": list(clean_updates),
                "update_digests": update_digests,
            },
            indent=2,
        )
        + "\n"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "source_update_digests.json").write_text(
        json.dumps(update_digests, indent=2) + "\n"
    )

    print("[3/4] Executing attack matrix", flush=True)
    p4 = P4Config(RobustnessPaths(root, output_root=output_root))
    if args.aggregators:
        p4 = __import__("dataclasses").replace(
            p4,
            aggregation=__import__("dataclasses").replace(
                p4.aggregation, methods=tuple(args.aggregators)
            ),
        )
    harness = RobustnessHarness(p4)
    threshold = (
        None if args.clipping_threshold == "median" else float(args.clipping_threshold)
    )
    p4 = __import__("dataclasses").replace(
        p4,
        aggregation=__import__("dataclasses").replace(
            p4.aggregation, clipping_threshold=threshold
        ),
    )
    poisoned_updates, asr_loader = {}, None
    attack_scope = tuple(args.attacks) if args.attacks else None
    if args.poisoned_dir:
        real = ("label_flipping", "targeted_backdoor")
        for attack in real:
            if attack_scope is not None and attack not in attack_scope:
                continue
            needed = {
                client
                for fraction in p4.threat.malicious_fractions
                for client in select_malicious_clients(p4, fraction)
            }
            poisoned_updates[attack] = {}
            for client in sorted(needed):
                path = args.poisoned_dir / attack / f"{client}_update.pt"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{path} missing; run P4/run_p4_poisoned_updates.py first"
                    )
                poisoned_updates[attack][client] = load_checkpoint_cpu(path)
        if attack_scope is None or "targeted_backdoor" in attack_scope:
            triggered = triggered_evaluation_rows(
                valid_frame,
                p4.backdoor.target_label,
                args.asr_images,
                config.training.seed,
                config.dataset.labels,
            )
            asr_loader = DataLoader(
                CheXpertDataset(
                    triggered,
                    args.data_root,
                    config.dataset.labels,
                    config.model.image_size,
                    {
                        "size": p4.backdoor.trigger_size,
                        "location": p4.backdoor.trigger_location,
                        "intensity": p4.backdoor.trigger_intensity,
                    },
                ),
                batch_size=32,
                shuffle=False,
                num_workers=args.num_workers,
            )
    harness = RobustnessHarness(p4, poisoned_updates, attack_scope)
    settings = {
        "aggregators": list(p4.aggregation.methods),
        "clipping_threshold": "median" if threshold is None else threshold,
    }
    if args.poisoned_dir:
        settings["real_poisoning"] = True
        settings["attacks"] = list(attack_scope) if attack_scope else "all"
    settings_path = output_root / "run_settings.json"
    if settings_path.is_file() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError(
            f"{output_root} was started with different settings; use a new --output-root"
        )
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    conditions_dir = output_root / "conditions"
    all_conditions = [
        condition
        for condition in build_condition_registry(p4)
        if attack_scope is None or condition.attack_name in attack_scope
    ]
    pending = [
        condition
        for condition in all_conditions
        if not (
            conditions_dir / condition.condition_id / "artifact_manifest.json"
        ).is_file()
    ]
    clean_path = output_root / "clean_reference.csv"
    methods = len(p4.aggregation.methods)
    print(
        f"  {len(pending)} conditions pending, "
        f"{len(all_conditions) - len(pending)} already complete",
        flush=True,
    )
    evaluations = (len(pending) + (0 if clean_path.is_file() else 1)) * methods
    eval_progress = Progress(
        max(len(valid_loader) * evaluations, 1), args.progress_every
    )
    evaluator = _evaluate_factory(
        valid_loader,
        valid_frame,
        config,
        device,
        eval_progress,
        asr_loader,
        p4.backdoor.target_label,
        p4.backdoor.attack_success_threshold,
    )
    results = harness.run_matrix(
        clean_updates,
        sample_counts,
        global_state,
        local_datasets,
        evaluator,
    )
    results.to_csv(output_root / "condition_results.csv", index=False)

    if clean_path.is_file():
        print("[4/4] Clean reference already exists; skipping", flush=True)
        print(
            f"Wrote {len(results)} attack rows to {output_root / 'condition_results.csv'}",
            flush=True,
        )
        return

    print("[4/4] Writing clean references and summary", flush=True)
    clean_rows = []
    for method in p4.aggregation.methods:
        clean_condition = type("CleanCondition", (), {"attack_name": "clean"})()
        _, macro, _ = evaluator(
            update_to_state_dict(
                global_state,
                harness._aggregate(method, clean_updates, sample_counts).update,
            ),
            clean_condition,
        )
        clean_rows.append(
            {
                "attack_name": "clean",
                "malicious_fraction": 0.0,
                "aggregation_method": method,
                **macro.iloc[0].to_dict(),
            }
        )
    clean_frame = pd.DataFrame(clean_rows)
    clean_frame = pd.DataFrame(
        [
            {
                "attack_name": condition.attack_name,
                "malicious_fraction": condition.malicious_fraction,
                "aggregation_method": method,
                **clean_frame.loc[clean_frame.aggregation_method == method]
                .iloc[0]
                .drop(
                    labels=["attack_name", "malicious_fraction", "aggregation_method"]
                )
                .to_dict(),
            }
            for condition in build_condition_registry(p4)
            for method in p4.aggregation.methods
        ]
    )
    clean_frame.to_csv(clean_path, index=False)
    print(
        f"Wrote {len(results)} attack rows to {output_root / 'condition_results.csv'}",
        flush=True,
    )
    print(f"Source updates: {source_dir}", flush=True)


if __name__ == "__main__":
    main()
