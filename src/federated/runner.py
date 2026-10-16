"""Configuration-driven federated runner with portable P1 artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from src.blockchain.identity import Identity
from src.blockchain.permissioned_ledger import PermissionedLedger
from src.blockchain.secure_ledger import SecureLedger
from src.blockchain.transactions import update_digest
from src.config import BlockchainConfig, ExperimentConfig
from src.federated.aggregation import fedavg
from src.federated.client import train_client
from src.monitoring.artifact_registry import write_manifest
from src.monitoring.progress import Progress
from src.monitoring.runtime import RuntimeTracker
from src.models.serialization import cpu_state_dict
from src.reproducibility import (
    load_checkpoint_cpu,
    move_state_to_device,
    save_checkpoint,
)


def _atomic_save(state: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_checkpoint(state, temporary)
    temporary.replace(path)


def _load_identities(output: Path, clients: int, blockchain: BlockchainConfig):
    """Secure runs persist keys so a resumed run keeps verifying its own ledger."""
    names = [f"client_{index}" for index in range(clients)]
    if not blockchain.secure:
        return [Identity.generate(name) for name in names], {}
    path = output / "identities.json"
    if path.is_file():
        saved = json.loads(path.read_text())
    else:
        saved = {
            "clients": {n: Identity.generate(n).private_hex() for n in names},
            "validators": {
                n: Identity.generate(n).private_hex() for n in blockchain.validators
            },
        }
        path.write_text(json.dumps(saved, indent=2) + "\n")
        path.chmod(0o600)
    return (
        [Identity.from_private_hex(n, saved["clients"][n]) for n in names],
        {
            n: Identity.from_private_hex(n, saved["validators"][n])
            for n in blockchain.validators
        },
    )


def _build_ledger(blockchain: BlockchainConfig, identities, validator_ids, blocks=None):
    if not blockchain.secure:
        if blocks is None:
            return PermissionedLedger(blockchain.validators, blockchain.majority)
        return PermissionedLedger.from_blocks(
            blocks, blockchain.validators, blockchain.majority
        )
    registry = {
        "validators": {n: i.public_hex for n, i in validator_ids.items()},
        "clients": {str(k): i.public_hex for k, i in enumerate(identities)},
        "majority": blockchain.majority,
    }
    if blocks is None:
        return SecureLedger(
            registry["validators"],
            registry["clients"],
            blockchain.majority,
            validator_ids,
        )
    return SecureLedger.from_blocks(blocks, registry, validator_ids)


class FederatedRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        model_factory: Callable[[], torch.nn.Module],
        device: torch.device,
        initial_state: dict[str, torch.Tensor] | None = None,
        progress_every: int = 50,
        checkpoint_every: int = 200,
        aggregate_fn: Callable[..., tuple[dict[str, torch.Tensor], list[dict]]]
        | None = None,
    ):
        if config.runtime.require_cuda and device.type != "cuda":
            raise RuntimeError(
                "CUDA-designated run cannot execute on a non-CUDA device"
            )
        self.config = config
        self.model_factory = model_factory
        self.device = device
        self.initial_state = initial_state
        self.progress_every = progress_every
        self.checkpoint_every = checkpoint_every
        self.aggregate_fn = aggregate_fn

    def run(
        self,
        client_loaders: list[Any],
        validation_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
        criterion: torch.nn.Module,
        run_id: str,
        resume: bool = False,
        eval_batches: int = 0,
    ) -> Path:
        """validation_fn(model, round_id, on_batch) -> (per_label, macro)."""
        if len(client_loaders) != self.config.federated.clients:
            raise ValueError("client loader count does not match frozen configuration")
        output = self.config.paths.output_dir / run_id
        checkpoint_path = output / "resume_checkpoint.pt"
        if output.exists() and any(output.iterdir()):
            if not resume:
                raise FileExistsError(
                    f"Refusing to overwrite existing run directory: {output} "
                    "(use resume to continue it)"
                )
        output.mkdir(parents=True, exist_ok=True)
        self.config.save(output / "training_config.json")
        blockchain = self.config.blockchain
        identities, validator_ids = _load_identities(
            output, len(client_loaders), blockchain
        )
        tracker = RuntimeTracker()
        global_model = self.model_factory().to(self.device)
        extra_rows: list[dict] = []
        history, client_rows, commitments = [], [], []
        start_round = 1
        if resume and checkpoint_path.is_file():
            saved = load_checkpoint_cpu(checkpoint_path)
            global_model.load_state_dict(saved["model_state"])
            meta = json.loads(saved["meta_json"])
            history, client_rows = meta["history"], meta["client_rows"]
            extra_rows = meta.get("extra", [])
            commitments, tracker.records = meta["commitments"], meta["runtime"]
            ledger = _build_ledger(
                blockchain, identities, validator_ids, meta["ledger"]
            )
            start_round = int(saved["round"]) + 1
            print(f"Resuming {run_id} after round {start_round - 1}", flush=True)
        else:
            if self.initial_state is not None:
                global_model.load_state_dict(self.initial_state)
            ledger = _build_ledger(blockchain, identities, validator_ids)
        total_rounds = self.config.federated.rounds
        if start_round > total_rounds:
            raise ValueError(f"{run_id} already completed {total_rounds} rounds")
        epochs = self.config.training.local_epochs
        client_batches = sum(len(loader) for loader in client_loaders) * epochs
        progress = Progress(
            (total_rounds - start_round + 1) * (client_batches + eval_batches),
            self.progress_every,
        )
        for round_id in range(start_round, total_rounds + 1):
            global_state = {
                key: value.detach().cpu().clone()
                for key, value in global_model.state_dict().items()
            }
            states, sizes = [], []
            partial = output / "partial"
            partial.mkdir(exist_ok=True)
            for stale in partial.iterdir():
                if (
                    not stale.name.startswith(f"round_{round_id:03d}_")
                    and stale.name != "mid.pt"
                ):
                    stale.unlink()
            for client_id, loader in enumerate(client_loaders):
                steps = len(loader) * epochs
                client_file = partial / f"round_{round_id:03d}_client_{client_id}.pt"
                mid_path = partial / "mid.pt"
                if client_file.is_file():
                    done = load_checkpoint_cpu(client_file)
                    snapshot = json.loads(done["meta_json"])
                    client_rows, commitments = (
                        snapshot["client_rows"],
                        snapshot["commitments"],
                    )
                    tracker.records = snapshot["runtime"]
                    ledger = _build_ledger(
                        blockchain, identities, validator_ids, snapshot["ledger"]
                    )
                    states.append(done["state"])
                    sizes.append(int(done["size"]))
                    progress.advance(steps)
                    print(
                        f"[{run_id}] r{round_id} client {client_id + 1} restored from disk",
                        flush=True,
                    )
                    continue
                generator = getattr(loader, "generator", None)
                if generator is not None:
                    generator.manual_seed(
                        self.config.training.seed + 1000 * round_id + client_id
                    )
                local_model = self.model_factory().to(self.device)
                local_model.load_state_dict(global_state)
                optimizer = torch.optim.AdamW(
                    local_model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                )
                start_batch, start_loss = 0, 0.0
                if mid_path.is_file():
                    mid = load_checkpoint_cpu(mid_path)
                    if mid["round"] == round_id and mid["client"] == client_id:
                        local_model.load_state_dict(mid["model_state"])
                        optimizer.load_state_dict(mid["optimizer_state"])
                        start_batch, start_loss = (
                            int(mid["batch"]),
                            float(mid["loss_sum"]),
                        )
                        progress.advance(start_batch)
                        print(
                            f"[{run_id}] r{round_id} client {client_id + 1} resuming at batch {start_batch}/{steps}",
                            flush=True,
                        )
                tag = f"{run_id} r{round_id}/{total_rounds} c{client_id + 1}/{len(client_loaders)}"

                def on_batch(batch: int, batch_loss: float, images: int) -> None:
                    progress.advance(1, images)
                    progress.tick(
                        tag, f"batch {batch}/{steps} loss {batch_loss:.4f}", batch
                    )

                def on_checkpoint(batch: int, loss_sum: float) -> None:
                    with tracker.measure(
                        "checkpoint_time", round=round_id, client_id=client_id
                    ):
                        _atomic_save(
                            {
                                "round": round_id,
                                "client": client_id,
                                "batch": batch,
                                "loss_sum": loss_sum,
                                "model_state": cpu_state_dict(local_model),
                                "optimizer_state": move_state_to_device(
                                    optimizer.state_dict(), torch.device("cpu")
                                ),
                            },
                            mid_path,
                        )

                with tracker.measure(
                    "training_time", round=round_id, client_id=client_id
                ):
                    state, loss, batches = train_client(
                        local_model,
                        loader,
                        criterion,
                        optimizer,
                        self.device,
                        self.config.training.local_epochs,
                        global_state,
                        self.config.federated.fedprox_mu
                        if self.config.federated.method == "FedProx"
                        else 0.0,
                        on_batch=on_batch,
                        start_batch=start_batch,
                        start_loss=start_loss,
                        checkpoint_every=self.checkpoint_every,
                        on_checkpoint=on_checkpoint,
                    )
                mid_path.unlink(missing_ok=True)
                with tracker.measure(
                    "signing_time", round=round_id, client_id=client_id
                ):
                    digest = update_digest(state)
                    signature = ledger.sign(
                        identities[client_id], round_id, client_id, digest
                    )
                with tracker.measure(
                    "consensus_time", round=round_id, client_id=client_id
                ):
                    ledger.record_update(round_id, client_id, digest, signature)
                states.append(state)
                sizes.append(len(loader.dataset))
                commitments.append(
                    {
                        "round_id": round_id,
                        "client_id": client_id,
                        "update_digest": digest,
                    }
                )
                client_rows.append(
                    {
                        "round": round_id,
                        "client_id": client_id,
                        "loss": loss,
                        "batches": batches,
                    }
                )
                _atomic_save(
                    {
                        "round": round_id,
                        "client": client_id,
                        "state": state,
                        "size": sizes[-1],
                        "meta_json": json.dumps(
                            {
                                "client_rows": client_rows,
                                "commitments": commitments,
                                "runtime": tracker.records,
                                "ledger": ledger.to_blocks(),
                            },
                            default=list,
                        ),
                    },
                    client_file,
                )
            with tracker.measure("aggregation_time", round=round_id):
                if self.aggregate_fn is None:
                    aggregated = fedavg(states, sizes)
                else:
                    aggregated, rows = self.aggregate_fn(
                        states, sizes, global_state, round_id
                    )
                    extra_rows.extend(rows)
                global_model.load_state_dict(aggregated)

            def on_eval_batch(images: int = 0) -> None:
                progress.advance(1, images)
                progress.tick(
                    f"{run_id} r{round_id}/{total_rounds} eval",
                    "validation",
                    progress.done,
                )

            with tracker.measure("evaluation_time", round=round_id):
                per_label, macro = validation_fn(global_model, round_id, on_eval_batch)
            row = {"round": round_id, **{k: float(v) for k, v in macro.iloc[0].items()}}
            history.append(row)
            print(
                f"[{run_id}] round {round_id}/{total_rounds} done: "
                f"macro AUROC {row['macro_auroc']:.4f}, macro AUPRC {row['macro_auprc']:.4f}",
                flush=True,
            )
            if blockchain.secure:
                with tracker.measure("commitment_time", round=round_id):
                    ledger.record_round(
                        round_id,
                        {
                            "global_input_digest": update_digest(global_state),
                            "global_output_digest": update_digest(
                                cpu_state_dict(global_model)
                            ),
                            "config_digest": hashlib.sha256(
                                (output / "training_config.json").read_bytes()
                            ).hexdigest(),
                            "metrics_digest": hashlib.sha256(
                                json.dumps(row, sort_keys=True).encode()
                            ).hexdigest(),
                            "client_update_digests": [
                                c["update_digest"]
                                for c in commitments
                                if c["round_id"] == round_id
                            ],
                        },
                    )
            ledger.save(output / "ledger.json")
            meta = {
                "history": history,
                "client_rows": client_rows,
                "commitments": commitments,
                "runtime": tracker.records,
                "ledger": ledger.to_blocks(),
                "extra": extra_rows,
            }
            temporary = output / "resume_checkpoint.pt.tmp"
            save_checkpoint(
                {
                    "round": round_id,
                    "model_state": cpu_state_dict(global_model),
                    "meta_json": json.dumps(meta, default=list),
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
            shutil.rmtree(output / "partial", ignore_errors=True)
        if not ledger.validate():
            raise RuntimeError("Ledger validation failed at end of run")
        save_checkpoint(cpu_state_dict(global_model), output / "endpoint_model.pt")
        pd.DataFrame(history).to_csv(output / "round_history.csv", index=False)
        pd.DataFrame(client_rows).to_csv(
            output / "client_round_details.csv", index=False
        )
        pd.DataFrame(commitments).to_csv(
            output / "blockchain_commitment_summary.csv", index=False
        )
        tracker.save(output / "runtime_breakdown.csv")
        if extra_rows:
            pd.DataFrame(extra_rows).to_csv(
                output / "aggregation_details.csv", index=False
            )
        (output / "environment_report.json").write_text(
            json.dumps(
                {"device_type": self.device.type, "torch": torch.__version__}, indent=2
            )
            + "\n"
        )
        per_label.to_csv(output / "endpoint_per_label_metrics.csv", index=False)
        macro.to_csv(output / "endpoint_macro_metrics.csv", index=False)
        # The validation callback owns patient-bearing prediction export.
        if not (output / "endpoint_valid_predictions.csv").exists():
            raise RuntimeError(
                "validation_fn must write endpoint_valid_predictions.csv before completion"
            )
        write_manifest(output, run_id, {"device_type": self.device.type})
        return output
