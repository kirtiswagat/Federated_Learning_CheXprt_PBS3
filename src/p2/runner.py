"""Generic P2 runner for scalability, subsampling, FedProx, and uncertainty studies."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from src.config import ExperimentConfig
from src.federated.aggregation import fedavg
from src.federated.client import train_client
from src.monitoring.artifact_registry import write_manifest
from src.monitoring.runtime import RuntimeTracker
from src.p2.registry import P2RunSpec
from src.reproducibility import load_checkpoint_cpu, save_checkpoint, seed_everything


ValidationResult = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


class P2Runner:
    """Run a fixed P2 specification using caller-provided existing data adapters.

    ``validation_fn`` must return ``(per_label_metrics, macro_metrics,
    patient_predictions)``. The runner deliberately does not implement a second
    dataset or evaluation pipeline.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        spec: P2RunSpec,
        model_factory: Callable[[], torch.nn.Module],
        device: torch.device,
    ):
        self.config = config
        self.spec = spec
        self.model_factory = model_factory
        self.device = device
        if config.runtime.require_cuda and device.type != "cuda":
            raise RuntimeError("CUDA-designated P2 run cannot execute on this device")

    @staticmethod
    def _model_bytes(model: torch.nn.Module) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )

    def run(
        self,
        client_loaders: list[Any],
        validation_fn: Callable[[torch.nn.Module, Path], ValidationResult],
        criterion: torch.nn.Module,
        run_id: str,
        initial_checkpoint: Path | None = None,
        resume_checkpoint: Path | None = None,
    ) -> Path:
        if len(client_loaders) != self.spec.clients:
            raise ValueError("client loader count does not match P2 specification")
        output = self.config.paths.root / "outputs" / "p2_scaling_sensitivity" / run_id
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing P2 run: {output}")
        output.mkdir(parents=True, exist_ok=False)
        seed_everything(self.config.training.seed)
        self.config.save(output / "training_config.json")
        model = self.model_factory().to(self.device)
        if initial_checkpoint:
            model.load_state_dict(load_checkpoint_cpu(initial_checkpoint))
        initial_state = copy.deepcopy(model.state_dict())
        start_round = 0
        history: list[dict[str, Any]] = []
        participation: list[dict[str, Any]] = []
        if resume_checkpoint:
            checkpoint = load_checkpoint_cpu(resume_checkpoint)
            model.load_state_dict(checkpoint["model_state"])
            start_round = int(checkpoint["round"])
            history = list(checkpoint.get("history", []))
        tracker = RuntimeTracker()
        rng = random.Random(self.config.training.seed)
        model_bytes = self._model_bytes(model)
        final_per_label = final_macro = final_predictions = None
        for round_id in range(start_round + 1, self.spec.rounds + 1):
            selected = list(range(self.spec.clients))
            if self.spec.participation_rate < 1:
                selected = sorted(rng.sample(selected, self.spec.selected_clients))
            global_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            states, sizes = [], []
            processed = 0
            losses = []
            round_start = time.perf_counter()
            for client_id in selected:
                local_model = self.model_factory().to(self.device)
                local_model.load_state_dict(global_state)
                optimizer = torch.optim.AdamW(
                    local_model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                )
                state, loss, _ = train_client(
                    local_model,
                    client_loaders[client_id],
                    criterion,
                    optimizer,
                    self.device,
                    self.config.training.local_epochs,
                    global_state,
                    self.spec.mu if self.spec.algorithm == "FedProx" else 0.0,
                )
                states.append(state)
                sizes.append(len(client_loaders[client_id].dataset))
                processed += (
                    len(client_loaders[client_id].dataset)
                    * self.config.training.local_epochs
                )
                losses.append(loss)
                participation.append(
                    {"round": round_id, "client_id": client_id, "selected": True}
                )
            for client_id in set(range(self.spec.clients)) - set(selected):
                participation.append(
                    {"round": round_id, "client_id": client_id, "selected": False}
                )
            model.load_state_dict(fedavg(states, sizes))
            per_label, macro, predictions = validation_fn(model, output)
            final_per_label, final_macro, final_predictions = (
                per_label,
                macro,
                predictions,
            )
            macro_row = macro.iloc[0].to_dict()
            history.append(
                {
                    "round": round_id,
                    "train_loss": sum(losses) / max(1, len(losses)),
                    "runtime_seconds": time.perf_counter() - round_start,
                    "processed_examples": processed,
                    "communication_bytes": model_bytes * len(selected) * 2,
                    "selected_clients": len(selected),
                    **macro_row,
                }
            )
            save_checkpoint(
                {
                    "round": round_id,
                    "model_state": model.state_dict(),
                    "history": history,
                },
                output / "resume_checkpoint.pt",
            )
        if final_per_label is None or final_macro is None or final_predictions is None:
            raise RuntimeError("P2 run produced no endpoint evaluation")
        save_checkpoint(model.state_dict(), output / "endpoint_model.pt")
        save_checkpoint(model.state_dict(), output / "checkpoint.pt")
        pd.DataFrame(history).to_csv(output / "round_history.csv", index=False)
        final_per_label.to_csv(output / "per_label_metrics.csv", index=False)
        final_macro.to_csv(output / "macro_metrics.csv", index=False)
        final_predictions.to_csv(output / "predictions.csv", index=False)
        pd.DataFrame(participation).to_csv(
            output / "client_participation.csv", index=False
        )
        history_frame = pd.DataFrame(history)
        history_frame.assign(operation="round_total").rename(
            columns={"runtime_seconds": "seconds"}
        )[["operation", "seconds"]].to_csv(output / "runtime_summary.csv", index=False)
        summary = {
            "run_id": run_id,
            "algorithm": self.spec.algorithm,
            "clients": self.spec.clients,
            "rounds": self.spec.rounds,
            "participation_rate": self.spec.participation_rate,
            "mu": self.spec.mu,
            "uncertainty_policy": self.spec.uncertainty_policy,
            "total_processed_examples": int(history_frame.processed_examples.sum()),
            "total_communication_bytes": int(history_frame.communication_bytes.sum()),
            "final_metrics": final_macro.iloc[0].to_dict(),
        }
        (output / f"{run_id.lower()}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )
        (output / "environment_report.json").write_text(
            json.dumps(
                {"device_type": self.device.type, "torch": torch.__version__}, indent=2
            )
            + "\n"
        )
        write_manifest(
            output, run_id, {"device_type": self.device.type, "source_stage": "P2"}
        )
        return output
