"""Build a reproducibility manifest from discovered, checksummed artifacts."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


def build_reproducibility_manifest(inventory, output_path: Path) -> dict:
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "artifacts": inventory.to_dict(orient="records"),
        "source_artifact_count": len(inventory),
        "scientific_results_generated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
