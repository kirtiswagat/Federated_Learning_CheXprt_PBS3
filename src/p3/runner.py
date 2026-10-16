"""Reusable CBRA federated runner for R12-R19."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch

from src.config import ExperimentConfig
from src.federated.client import train_client
from src.monitoring.artifact_registry import write_manifest
from src.p3.cbra import cbra_aggregate, compute_contributions
from src.p3.registry import P3RunSpec
from src.reproducibility import load_checkpoint_cpu, save_checkpoint, seed_everything


class P3Runner:
    """Execute CBRA without duplicating the existing dataset/evaluation pipeline."""

    def __init__(
        self,
        config: ExperimentConfig,
        spec: P3RunSpec,
        model_factory: Callable[[], torch.nn.Module],
        device: torch.device,
    ):
        self.config = config
        self.spec = spec
        self.model_factory = model_factory
        self.device = device

    def run(
        self,
        client_loaders: Sequence[Any],
        client_labels: Sequence[np.ndarray],
        global_labels: np.ndarray,
        validation_fn: Callable[
            [torch.nn.Module, Path], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ],
        criterion: torch.nn.Module,
        run_id: str,
        initial_checkpoint: Path | None = None,
        resume_checkpoint: Path | None = None,
    ) -> Path:
        if (
            len(client_loaders) != self.spec.clients
            or len(client_labels) != self.spec.clients
        ):
            raise ValueError(
                "P3 client loaders and label arrays must match the five-client specification"
            )
        output = self.config.paths.root / "outputs" / "p3_cbra" / run_id
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing P3 run: {output}")
        output.mkdir(parents=True, exist_ok=False)
        seed_everything(self.config.training.seed)
        self.config.save(output / "training_config.json")
        model = self.model_factory().to(self.device)
        if initial_checkpoint:
            model.load_state_dict(load_checkpoint_cpu(initial_checkpoint))
        start_round = 0
        history: list[dict[str, Any]] = []
        contribution_rows: list[pd.DataFrame] = []
        if resume_checkpoint:
            checkpoint = load_checkpoint_cpu(resume_checkpoint)
            model.load_state_dict(checkpoint["model_state"])
            start_round = int(checkpoint["round"])
            history = list(checkpoint.get("history", []))
        final_per_label = final_macro = final_predictions = None
        for round_id in range(start_round + 1, self.spec.rounds + 1):
            global_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            states, sizes, losses = [], [], []
            for loader in client_loaders:
                local_model = self.model_factory().to(self.device)
                local_model.load_state_dict(global_state)
                optimizer = torch.optim.AdamW(
                    local_model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                )
                state, loss, _ = train_client(
                    local_model,
                    loader,
                    criterion,
                    optimizer,
                    self.device,
                    self.config.training.local_epochs,
                    global_state,
                    0.0,
                )
                states.append(state)
                sizes.append(len(loader.dataset))
                losses.append(loss)
            contributions = compute_contributions(
                states,
                global_state,
                sizes,
                client_labels,
                global_labels,
                self.spec.alpha_n,
                self.spec.beta_q,
                self.spec.gamma_d,
            )
            contributions.insert(0, "round", round_id)
            contribution_rows.append(contributions)
            model.load_state_dict(cbra_aggregate(states, contributions))
            final_per_label, final_macro, final_predictions = validation_fn(
                model, output
            )
            history.append(
                {
                    "round": round_id,
                    "train_loss": float(np.mean(losses)),
                    **final_macro.iloc[0].to_dict(),
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
            raise RuntimeError("P3 run produced no endpoint evaluation")
        save_checkpoint(model.state_dict(), output / "checkpoint.pt")
        save_checkpoint(model.state_dict(), output / "endpoint_model.pt")
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(output / "round_history.csv", index=False)
        final_per_label.to_csv(output / "per_label_metrics.csv", index=False)
        final_macro.to_csv(output / "macro_metrics.csv", index=False)
        final_predictions.to_csv(output / "best_valid_predictions.csv", index=False)
        contributions_frame = pd.concat(contribution_rows, ignore_index=True)
        contributions_frame.to_csv(output / "cbra_round_details.csv", index=False)
        contributions_frame.to_csv(output / "cbra_weights.csv", index=False)
        history_frame.to_csv(output / "training_summary.csv", index=False)
        (output / "environment_report.json").write_text(
            json.dumps(
                {"device_type": self.device.type, "torch": torch.__version__}, indent=2
            )
            + "\n"
        )
        summary = {
            "run_id": run_id,
            "variant": self.spec.variant,
            "alpha_n": self.spec.alpha_n,
            "beta_q": self.spec.beta_q,
            "gamma_d": self.spec.gamma_d,
            "final_metrics": final_macro.iloc[0].to_dict(),
        }
        (output / f"{run_id.lower()}_results.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )
        write_manifest(
            output, run_id, {"device_type": self.device.type, "source_stage": "P3"}
        )
        return output
