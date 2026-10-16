"""Machine-readable P4 adversary assumptions."""

from __future__ import annotations

import json
from pathlib import Path

THREAT_MODEL = {
    "clients": "simulated registered federated participants from one source dataset",
    "malicious_capabilities": [
        "modify local labels",
        "modify local training",
        "modify submitted updates",
    ],
    "honest_data_protected": True,
    "attacked_tensors": "trainable parameters only; BatchNorm running statistics and integer counters are not modified",
    "validation_cohort_protected": True,
    "frozen_global_input_protected": True,
    "signature_forgery_allowed": False,
    "server_knows_malicious_clients": False,
    "scope": "controlled one-round update robustness stress test",
    "claims_excluded": [
        "complete attack resistance",
        "production Byzantine fault tolerance",
        "blockchain prevents poisoning",
    ],
}


def save_threat_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(THREAT_MODEL, indent=2, sort_keys=True) + "\n")


def threat_model_markdown() -> str:
    return """# P4 Threat Model\n\nClients are simulated registered participants from one source dataset. A malicious client may alter its local labels, training, or submitted update, but cannot alter honest data, the validation cohort, the frozen global input checkpoint, or another client's signature. This is a controlled one-round stress test; auditability authenticates origin and integrity but does not prove update safety.\n"""
