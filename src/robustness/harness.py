"""Fair one-round P4 condition-matrix harness."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.monitoring.artifact_registry import write_manifest
from src.robustness.aggregators import (
    aggregate_coordinate_median,
    aggregate_fedavg,
    aggregate_norm_clipped_fedavg,
    aggregate_trimmed_mean,
)
from src.robustness.attacks import (
    ByzantineRandomAttack,
    LabelFlippingAttack,
    ModelReplacementAttack,
    SignFlippingAttack,
    TargetedBackdoorAttack,
)
from src.robustness.attacks.base import AttackResult
from src.robustness.config import P4Config, RobustnessCondition
from src.robustness.registry import build_condition_registry, select_malicious_clients
from src.robustness.threat_model import (
    THREAT_MODEL,
    save_threat_model,
    threat_model_markdown,
)
from src.robustness.update_utils import (
    TensorUpdate,
    compute_update_digest,
    compute_update_l2_norm,
    update_to_state_dict,
)

ATTACK_IMPLEMENTATIONS = {
    "label_flipping": LabelFlippingAttack,
    "sign_flipping": SignFlippingAttack,
    "model_replacement": ModelReplacementAttack,
    "targeted_backdoor": TargetedBackdoorAttack,
    "byzantine_random": ByzantineRandomAttack,
}


class ProvidedUpdateAttack:
    """Submits an update the malicious client trained on poisoned data."""

    def __init__(self, name: str, updates: Mapping[str, TensorUpdate]):
        self.name = name
        self.updates = updates

    def apply(
        self,
        client_id,
        clean_update,
        local_dataset,
        global_state,
        attack_config,
        random_generator,
    ):
        if client_id not in self.updates:
            raise KeyError(f"No poisoned update for {self.name}/{client_id}")
        submitted = {
            k: v.detach().cpu().clone() for k, v in self.updates[client_id].items()
        }
        return AttackResult(
            self.name,
            str(client_id),
            True,
            submitted,
            compute_update_digest(clean_update),
            compute_update_digest(submitted),
            compute_update_l2_norm(clean_update),
            compute_update_l2_norm(submitted),
            {"source": "retrained_on_poisoned_data"},
            0,
            {},
            "valid",
        )


class RobustnessHarness:
    def __init__(
        self,
        config: P4Config,
        poisoned_updates: Mapping[str, Mapping[str, TensorUpdate]] | None = None,
        attack_names: tuple[str, ...] | None = None,
    ):
        self.config = config
        self.poisoned_updates = poisoned_updates or {}
        self.attack_names = attack_names

    def _aggregate(
        self,
        method: str,
        updates: Mapping[str, TensorUpdate],
        sample_counts: Mapping[str, int],
    ):
        if method == "fedavg":
            return aggregate_fedavg(updates, sample_counts)
        if method == "coordinate_median":
            return aggregate_coordinate_median(updates, sample_counts)
        if method == "trimmed_mean":
            return aggregate_trimmed_mean(
                updates, sample_counts, self.config.aggregation.trimmed_mean_count
            )
        if method == "norm_clipped_fedavg":
            return aggregate_norm_clipped_fedavg(
                updates,
                sample_counts,
                self.config.aggregation.clipping_threshold,
                self.config.aggregation.clipping_epsilon,
            )
        raise ValueError(f"Unknown P4 aggregator: {method}")

    def run_condition(
        self,
        condition: RobustnessCondition,
        clean_updates: Mapping[str, TensorUpdate],
        sample_counts: Mapping[str, int],
        global_state: Mapping[str, Any],
        local_datasets: Mapping[str, Any] | None = None,
        evaluator: Callable[
            [Mapping[str, Any], RobustnessCondition], tuple[pd.DataFrame, pd.DataFrame]
        ]
        | None = None,
    ) -> pd.DataFrame:
        client_ids = list(clean_updates)
        if set(client_ids) != set(self.config.client_ids):
            raise ValueError(
                "clean updates must contain exactly the configured client IDs"
            )
        malicious = set(
            select_malicious_clients(self.config, condition.malicious_fraction)
        )
        attack = ATTACK_IMPLEMENTATIONS[condition.attack_name]()
        if condition.attack_name in self.poisoned_updates:
            attack = ProvidedUpdateAttack(
                condition.attack_name, self.poisoned_updates[condition.attack_name]
            )
        generator = __import__("torch").Generator().manual_seed(condition.seed)
        attacked_results = {}
        for client_id in client_ids:
            if client_id in malicious:
                attacked_results[client_id] = attack.apply(
                    client_id,
                    clean_updates[client_id],
                    (local_datasets or {}).get(client_id),
                    global_state,
                    self.config.attack
                    if condition.attack_name != "targeted_backdoor"
                    else self.config.backdoor,
                    generator,
                )
            else:
                digest = compute_update_digest(clean_updates[client_id])
                attacked_results[client_id] = type(
                    "CleanResult",
                    (),
                    {
                        "attack_name": "clean",
                        "client_id": client_id,
                        "malicious": False,
                        "submitted_update": copy.deepcopy(clean_updates[client_id]),
                        "original_digest": digest,
                        "attacked_digest": digest,
                        "original_norm": compute_update_l2_norm(
                            clean_updates[client_id]
                        ),
                        "attacked_norm": compute_update_l2_norm(
                            clean_updates[client_id]
                        ),
                        "attack_parameters": {},
                        "modified_sample_count": 0,
                        "metadata": {},
                        "validation_status": "valid",
                    },
                )()
        attacked_updates = {
            client: result.submitted_update
            for client, result in attacked_results.items()
        }
        update_digest = hashlib.sha256(
            "".join(
                compute_update_digest(attacked_updates[client])
                for client in sorted(client_ids)
            ).encode()
        ).hexdigest()
        condition_dir = (
            self.config.paths.output_root / "conditions" / condition.condition_id
        )
        condition_dir.mkdir(parents=True, exist_ok=False)
        condition_config = {
            **asdict(condition),
            "malicious_client_count": len(malicious),
            "malicious_client_ids": sorted(malicious),
            "update_set_digest": update_digest,
        }
        (condition_dir / "condition_config.json").write_text(
            json.dumps(condition_config, indent=2) + "\n"
        )
        save_threat_model(condition_dir / "threat_model.json")
        (condition_dir / "threat_model.md").write_text(threat_model_markdown())
        pd.DataFrame(
            [
                {"client_id": client, "malicious": client in malicious}
                for client in client_ids
            ]
        ).to_csv(condition_dir / "malicious_client_manifest.csv", index=False)
        pd.DataFrame(
            [
                {
                    **{
                        "client_id": client,
                        "malicious": result.malicious,
                        "original_digest": result.original_digest,
                        "attacked_digest": result.attacked_digest,
                        "original_norm": result.original_norm,
                        "attacked_norm": result.attacked_norm,
                        "modified_sample_count": result.modified_sample_count,
                    },
                    **result.metadata,
                }
                for client, result in attacked_results.items()
            ]
        ).to_csv(condition_dir / "client_update_statistics.csv", index=False)
        attack_details = pd.DataFrame(
            [
                {
                    "client_id": client,
                    "attack_name": result.attack_name,
                    "malicious": result.malicious,
                    "attack_parameters": json.dumps(
                        result.attack_parameters, sort_keys=True
                    ),
                    "modified_sample_count": result.modified_sample_count,
                    "validation_status": result.validation_status,
                }
                for client, result in attacked_results.items()
            ]
        )
        attack_details.to_csv(condition_dir / "attack_details.csv", index=False)
        pd.DataFrame(
            [
                {
                    "client_id": client,
                    "submitted_update_digest": result.attacked_digest,
                    "signature_verification": "not_configured",
                }
                for client, result in attacked_results.items()
            ]
        ).to_csv(condition_dir / "client_update_digests.csv", index=False)
        rows = []
        per_label_frames = []
        prediction_frames = []
        for method in self.config.aggregation.methods:
            result = self._aggregate(method, attacked_updates, sample_counts)
            metrics = {}
            if evaluator:
                evaluated = evaluator(
                    update_to_state_dict(global_state, result.update), condition
                )
                if len(evaluated) == 3:
                    per_label, macro, predictions = evaluated
                    per_label = per_label.copy()
                    per_label.insert(0, "aggregation_method", method)
                    per_label_frames.append(per_label)
                    predictions = predictions.copy()
                    predictions.insert(0, "aggregation_method", method)
                    prediction_frames.append(predictions)
                else:
                    _, macro = evaluated
                metrics = macro.iloc[0].to_dict()
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "attack_name": condition.attack_name,
                    "malicious_fraction": condition.malicious_fraction,
                    "aggregation_method": method,
                    "update_set_digest": update_digest,
                    "global_input_digest": compute_update_digest(
                        {
                            key: value
                            for key, value in global_state.items()
                            if hasattr(value, "shape")
                        }
                    ),
                    "aggregation_seconds": result.elapsed_seconds,
                    **metrics,
                }
            )
        result_frame = pd.DataFrame(rows)
        result_frame.to_csv(condition_dir / "aggregation_details.csv", index=False)
        result_frame.to_csv(condition_dir / "macro_metrics.csv", index=False)
        (
            pd.concat(per_label_frames, ignore_index=True)
            if per_label_frames
            else pd.DataFrame()
        ).to_csv(condition_dir / "per_label_metrics.csv", index=False)
        (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        ).to_csv(condition_dir / "attacked_validation_predictions.csv", index=False)
        pd.DataFrame().to_csv(
            condition_dir / "clean_validation_predictions.csv", index=False
        )
        pd.DataFrame().to_csv(
            condition_dir / "attack_specific_metrics.csv", index=False
        )
        pd.DataFrame(
            [
                {"operation": "aggregation", "seconds": row["aggregation_seconds"]}
                for row in rows
            ]
        ).to_csv(condition_dir / "runtime_events.csv", index=False)
        pd.DataFrame(
            [
                {
                    "condition_id": condition.condition_id,
                    "ledger_valid": False,
                    "signature_valid": False,
                    "provenance_status": "not_configured",
                    "note": "Enable P1 signing/ledger bridge for measured commitments.",
                }
            ]
        ).to_csv(condition_dir / "ledger_commitment_summary.csv", index=False)
        (condition_dir / "fairness_checksum_report.json").write_text(
            json.dumps(
                {
                    "update_set_digest": update_digest,
                    "aggregators_received_identical_updates": True,
                    "malicious_clients": sorted(malicious),
                },
                indent=2,
            )
            + "\n"
        )
        write_manifest(
            condition_dir,
            condition.condition_id,
            {"source_stage": "P4", "update_set_digest": update_digest},
        )
        return result_frame

    def run_matrix(
        self,
        clean_updates: Mapping[str, TensorUpdate],
        sample_counts: Mapping[str, int],
        global_state: Mapping[str, Any],
        local_datasets: Mapping[str, Any] | None = None,
        evaluator=None,
    ) -> pd.DataFrame:
        conditions = [
            condition
            for condition in build_condition_registry(self.config)
            if self.attack_names is None or condition.attack_name in self.attack_names
        ]
        frames = []
        for index, condition in enumerate(conditions, start=1):
            condition_dir = (
                self.config.paths.output_root / "conditions" / condition.condition_id
            )
            if (condition_dir / "artifact_manifest.json").is_file():
                frames.append(pd.read_csv(condition_dir / "macro_metrics.csv"))
                print(
                    f"[P4] condition {index}/{len(conditions)} already complete, "
                    f"skipping: {condition.condition_id}",
                    flush=True,
                )
                continue
            if condition_dir.exists():
                shutil.rmtree(condition_dir)  # interrupted before its manifest
            print(
                f"[P4] condition {index}/{len(conditions)}: {condition.condition_id}",
                flush=True,
            )
            frame = self.run_condition(
                condition,
                clean_updates,
                sample_counts,
                global_state,
                local_datasets,
                evaluator,
            )
            frames.append(frame)
            print(
                f"[P4] completed {index}/{len(conditions)}: {condition.condition_id}",
                flush=True,
            )
        return pd.concat(frames, ignore_index=True)
