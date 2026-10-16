"""Runtime breakdown, real-size blockchain overhead and a data-derived runtime note."""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.blockchain.identity import Identity
from src.blockchain.overhead import benchmark
from src.blockchain.permissioned_ledger import PermissionedLedger
from src.blockchain.transactions import update_digest
from src.reproducibility import load_checkpoint_cpu

RUNS = ("R01", "R02", "R03", "R04", "R05")
PARTS = (
    "training_time",
    "evaluation_time",
    "aggregation_time",
    "signing_time",
    "consensus_time",
    "checkpoint_time",
)


def runtime_summary(runs: Path, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, per_round = [], []
    for run in RUNS:
        records = pd.read_csv(runs / run / "runtime_breakdown.csv")
        details = pd.read_csv(runs / run / "client_round_details.csv")
        seconds = records.groupby("operation").seconds.sum()
        get = lambda name: float(seconds.get(name, 0.0))
        net_train = get("training_time") - get("checkpoint_time")
        total = net_train + sum(get(p) for p in PARTS if p != "training_time")
        images = int(details.batches.sum()) * batch_size
        row = {
            "run": run,
            "rounds": int(details["round"].nunique()),
            "approx_train_images": images,
            "train_seconds_net": net_train,
            **{f"{p}_s": get(p) for p in PARTS},
            "total_accounted_s": total,
            "train_img_per_s": images / net_train if net_train else float("nan"),
        }
        for part in PARTS:
            value = net_train if part == "training_time" else get(part)
            row[f"{part}_share_pct"] = 100 * value / total
        row["ledger_share_pct"] = (
            row["signing_time_share_pct"] + row["consensus_time_share_pct"]
        )
        summary.append(row)
        train = records[records.operation == "training_time"]
        rounds = train.groupby("round").seconds.sum().reset_index()
        rounds.insert(0, "run", run)
        per_round.append(rounds.rename(columns={"seconds": "training_seconds"}))
    return pd.DataFrame(summary), pd.concat(per_round, ignore_index=True)


def real_update_overhead(runs: Path, repeats: int = 5) -> pd.DataFrame:
    state = load_checkpoint_cpu(runs / "R01" / "endpoint_model.pt")
    size_mb = sum(v.numel() * v.element_size() for v in state.values()) / 1e6
    identity = Identity.generate("overhead")
    rows = []
    for repeat in range(repeats):
        start = time.perf_counter()
        digest = update_digest(state)
        digest_s = time.perf_counter() - start
        start = time.perf_counter()
        signature = identity.sign(digest.encode())
        sign_s = time.perf_counter() - start
        start = time.perf_counter()
        identity.verify(digest.encode(), signature)
        verify_s = time.perf_counter() - start
        ledger = PermissionedLedger()
        start = time.perf_counter()
        ledger.append(1, digest)
        append_s = time.perf_counter() - start
        rows.append(
            {
                "repeat": repeat,
                "update_mb": size_mb,
                "digest_s": digest_s,
                "sign_s": sign_s,
                "verify_s": verify_s,
                "ledger_append_s": append_s,
            }
        )
    return pd.DataFrame(rows)


def write_note(path: Path, summary: pd.DataFrame, overhead: pd.DataFrame) -> None:
    lines = ["# Runtime root-cause note (generated from runtime_breakdown.csv)", ""]
    lines.append(
        "| Run | Train h | Eval h | Aggregation s | Ledger s | Ledger share % | Train img/s |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.run} | {r.train_seconds_net / 3600:.2f} | {r.evaluation_time_s / 3600:.2f} | "
            f"{r.aggregation_time_s:.1f} | {r.signing_time_s + r.consensus_time_s:.1f} | "
            f"{r.ledger_share_pct:.4f} | {r.train_img_per_s:.1f} |"
        )
    mean = overhead.mean(numeric_only=True)
    lines += [
        "",
        f"- Local training dominates wall-clock: {summary.training_time_share_pct.min():.1f} to "
        f"{summary.training_time_share_pct.max():.1f}% across runs; evaluation is "
        f"{summary.evaluation_time_share_pct.min():.1f} to {summary.evaluation_time_share_pct.max():.1f}%.",
        f"- Ledger work (signing plus consensus) is at most {summary.ledger_share_pct.max():.4f}% of accounted time.",
        f"- A real {mean.update_mb:.0f} MB update costs {mean.digest_s:.2f} s to digest, "
        f"{mean.sign_s * 1000:.2f} ms to sign, {mean.verify_s * 1000:.2f} ms to verify and "
        f"{mean.ledger_append_s * 1000:.2f} ms to append (mean of {len(overhead)} repeats).",
        "- Throughput is images per second of net training time (training minus mid-client checkpoint writes); "
        "image counts are batches x batch size, so the last partial batch is counted as full.",
        "- Runs R01-R05 were executed sequentially on one Apple MPS device; ledger cost is measured on the same machine.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    base = args.root / "Output" / "p1_critical_revision"
    out = base / "runtime"
    out.mkdir(parents=True, exist_ok=True)
    print("[1/4] Runtime breakdown", flush=True)
    summary, per_round = runtime_summary(base / "runs", args.batch_size)
    summary.to_csv(out / "runtime_summary.csv", index=False)
    per_round.to_csv(out / "training_seconds_per_round.csv", index=False)
    print("[2/4] Real-size update overhead", flush=True)
    overhead = real_update_overhead(base / "runs")
    overhead.to_csv(out / "real_update_overhead.csv", index=False)
    print("[3/4] Synthetic ledger scaling benchmark", flush=True)
    scaling, growth = benchmark()
    scaling.to_csv(out / "ledger_scaling_benchmark.csv", index=False)
    growth.to_csv(out / "ledger_growth.csv", index=False)
    print("[4/4] Note", flush=True)
    write_note(out / "runtime_root_cause_note.md", summary, overhead)
    print(
        summary[
            [
                "run",
                "train_seconds_net",
                "evaluation_time_s",
                "ledger_share_pct",
                "train_img_per_s",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"Written to {out}", flush=True)


if __name__ == "__main__":
    main()
