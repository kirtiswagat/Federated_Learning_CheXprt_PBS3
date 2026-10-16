"""Export the P4 threat-model assumptions as a CSV table and markdown."""

import argparse
from pathlib import Path

import pandas as pd

from src.robustness.threat_model import THREAT_MODEL, save_threat_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root / "threat_model"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, value in THREAT_MODEL.items():
        text = "; ".join(value) if isinstance(value, list) else str(value)
        rows.append({"assumption": key, "value": text})
    table = pd.DataFrame(rows)
    table.to_csv(out / "threat_model_table.csv", index=False)
    lines = ["| Assumption | Value |", "|---|---|"]
    lines += [f"| {r.assumption} | {r.value} |" for r in table.itertuples()]
    (out / "threat_model_table.md").write_text("\n".join(lines) + "\n")
    save_threat_model(out / "threat_model.json")
    print(table.to_string(index=False))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
