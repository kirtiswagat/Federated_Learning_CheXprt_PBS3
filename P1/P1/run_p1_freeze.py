"""Freeze the MPS environment and write the common initial checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path

from src.config import default_config, environment_report, run_configs
from src.device import resolve_mps
from src.models.resnet_multilabel import build_resnet50
from src.models.serialization import save_model
from src.reproducibility import seed_everything

PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "Pillow",
    "cryptography",
    "pytest",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = default_config(args.root)
    out = args.root / "Output" / "p1_critical_revision" / "frozen"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Checking MPS", flush=True)
    device = resolve_mps(base.runtime.require_mps)

    print("[2/4] Writing frozen config", flush=True)
    frozen = {
        "device": device.type,
        "environment": environment_report(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "base_config": base.to_dict(),
        "runs": {
            rid: cfg.federated.__dict__ for rid, cfg in run_configs(args.root).items()
        },
    }
    config_path = out / "frozen_config.json"
    config_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True, default=str) + "\n"
    )

    print("[3/4] Building common initial checkpoint", flush=True)
    seed_everything(base.training.seed)
    model = build_resnet50(
        base.model.num_labels, base.model.pretrained, base.model.dropout
    )
    checkpoint = out / "initial_checkpoint.pt"
    checkpoint_sha = save_model(model, checkpoint)

    print("[4/4] Writing checksums", flush=True)
    lines = [
        f"{_sha256(config_path)}  {config_path.name}",
        f"{checkpoint_sha}  {checkpoint.name}",
    ]
    (out / "frozen.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"Done: {out}", flush=True)


if __name__ == "__main__":
    main()
