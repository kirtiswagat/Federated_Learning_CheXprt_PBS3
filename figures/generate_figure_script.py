"""Generate the data block of make_figures.py from the evidence pack (run once; the result is standalone)."""

import pprint
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = (
    HERE.parents[1]
    if (HERE.parents[1] / "Manuscript_Findings_Evidence_Pack.md").exists()
    else HERE.parents[0]
)
PACK_PATH = (
    next(
        p
        for p in [HERE.parents[1], HERE.parents[0]]
        if (p / "Manuscript_Findings_Evidence_Pack.md").exists()
    )
    / "Manuscript_Findings_Evidence_Pack.md"
)
PACK = PACK_PATH.read_text().splitlines()

NAMES = {
    "R01 (round 5)": "FedAvg-IID, round 5",
    "R01 (round 10)": "FedAvg-IID, round 10",
    "R01_r5": "FedAvg-IID, round 5",
    "R01_r10": "FedAvg-IID, round 10",
    "R01": "FedAvg-IID",
    "R02": "FedAvg-Dir0.3",
    "R03": "FedAvg-Dir0.1",
    "R04": "FedProx-IID",
    "R05": "FedProx-Dir0.1",
    "R12": "CBRA-v1",
    "R13": "CBRA-v2",
    "R14": "sample-lower",
    "R15": "sample-higher",
    "R16": "rarity-lower",
    "R18": "drift-off",
    "R19": "drift-strong",
    "CBRA_v1": "CBRA-v1",
    "CBRA_v2": "CBRA-v2",
    "sample_lower": "sample-lower",
    "sample_higher": "sample-higher",
    "rarity_lower": "rarity-lower",
    "drift_disabled": "drift-off",
    "drift_strong": "drift-strong",
}


def table(anchor):
    start = next(i for i, line in enumerate(PACK) if re.search(anchor, line))
    i = start + 1
    while not PACK[i].startswith("|"):
        i += 1
    rows = []
    while i < len(PACK) and PACK[i].startswith("|"):
        cells = [c.strip() for c in PACK[i].strip().strip("|").split("|")]
        if not all(set(c) <= set("-: ") for c in cells):
            rows.append(cells)
        i += 1
    return rows


def ci(text):
    m = re.match(r"([-+]?\d*\.?\d+)\s*\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", text)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def nm(key):
    return NAMES.get(key, key)


data = {}

rows = table(r"^### 3\.1 Macro results")
data["MACRO"] = {
    nm(r[0]): {"auroc": ci(r[3]), "auprc": ci(r[4]), "auroc5": ci(r[5])}
    for r in rows[1:]
}

rows = table(r"^### 3\.3 Per-label AUROC")
runs = [nm(h) for h in rows[0][1:]]
data["PER_LABEL_AUROC"] = {
    r[0]: {run: ci(c) for run, c in zip(runs, r[1:])} for r in rows[1:]
}

rows = table(r"^### 6\.3 Results")
data["CBRA"] = {
    nm(r[1]): {
        "alpha_n": float(r[2]),
        "beta_q": float(r[3]),
        "gamma_d": float(r[4]),
        "auroc": ci(r[5]),
        "auprc": ci(r[6]),
        "diff_vs_fedavg": ci(r[7]),
    }
    for r in rows[1:]
}

rows = table(r"^Normalised aggregation weights")
data["CBRA_WEIGHTS"] = {
    nm(r[0]): [float(x) for x in r[1:]] for r in rows[1:] if r[0] != "R17"
}

rows = table(r"^AUROC drop = ")
data["ATTACK_DROP"] = {
    (r[0], float(r[1])): dict(zip(rows[0][2:], [float(x) for x in r[2:]]))
    for r in rows[1:]
}

rows = table(r"^Clean \(no attack\)")
data["CLEAN"] = {
    r[0]: {"auroc": float(r[1]), "auprc": float(r[2]), "asr": float(r[3])}
    for r in rows[1:]
}

rows = table(r"^Backdoor attack success rate")
data["ASR"] = [(float(r[0]), r[1], float(r[2]), float(r[3])) for r in rows[1:]]

rows = table(r"^### 4\.1 Measured runtime")
data["RUNTIME"] = [
    (nm(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[6]), float(r[7]))
    for r in rows[1:]
]

rows = table(r"^\*\*\(a\) How common")
data["UNCERTAIN"] = [(r[0], float(r[5]), float(r[6])) for r in rows[1:]]

rows = table(r"^Macro AUROC \(14 labels\) with 95%")
data["POLICY"] = {
    nm(r[0]): {"u_zero": ci(r[1]), "u_ignore": ci(r[2]), "u_one": ci(r[4])}
    for r in rows[1:]
}

block = "\n\n".join(
    f"{key} = {pprint.pformat(value, width=100, sort_dicts=False)}"
    for key, value in data.items()
)
template = (HERE / "make_figures_template.py").read_text()
(HERE / "make_figures.py").write_text(template.replace("# DATA_BLOCK", block))
print("written", len(block), "characters of data")
