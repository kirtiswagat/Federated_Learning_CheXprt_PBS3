"""Print a frozen P2 run specification for a caller-provided data adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict

from src.p2.registry import P2RunRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    revision_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output-root",
        type=Path,
        default=revision_root / "Output" / "p2_scaling_sensitivity",
    )
    args = parser.parse_args()
    print(f"[P2] Loading registry for {args.run_id}", flush=True)
    spec = P2RunRegistry(args.root).get(args.run_id)
    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[P2] Resolved specification: {spec}", flush=True)
    (output_dir / "run_spec.json").write_text(
        json.dumps(
            {
                **asdict(spec),
                "status": "configuration_ready",
                "training_adapter_required": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[P2] Run specification written to {output_dir / 'run_spec.json'}", flush=True
    )


if __name__ == "__main__":
    main()
