"""Print a frozen P3 CBRA specification for a caller-provided training adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict

from src.p3.registry import P3RunRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    revision_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output-root",
        type=Path,
        default=revision_root / "Output" / "p3_cbra",
    )
    args = parser.parse_args()
    print(f"[P3] Loading registry for {args.run_id}", flush=True)
    spec = P3RunRegistry(args.root).get(args.run_id)
    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[P3] Resolved specification: {spec}", flush=True)
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
        f"[P3] Run specification written to {output_dir / 'run_spec.json'}", flush=True
    )


if __name__ == "__main__":
    main()
