# Runtime root-cause note (generated from runtime_breakdown.csv)

| Run | Train h | Eval h | Aggregation s | Ledger s | Ledger share % | Train img/s |
|---|---|---|---|---|---|---|
| R01 | 8.58 | 0.64 | 1.0 | 2.9 | 0.0088 | 49.6 |
| R02 | 4.30 | 0.32 | 0.5 | 1.5 | 0.0089 | 49.4 |
| R03 | 4.34 | 0.32 | 0.6 | 1.5 | 0.0091 | 49.0 |
| R04 | 4.34 | 0.31 | 0.5 | 1.5 | 0.0087 | 49.0 |
| R05 | 4.33 | 0.31 | 0.5 | 1.5 | 0.0088 | 49.1 |

- Local training dominates wall-clock: 92.5 to 92.7% across runs; evaluation is 6.7 to 6.9%.
- Ledger work (signing plus consensus) is at most 0.0091% of accounted time.
- A real 94 MB update costs 0.06 s to digest, 0.08 ms to sign, 0.06 ms to verify and 0.03 ms to append (mean of 5 repeats).
- Throughput is images per second of net training time (training minus mid-client checkpoint writes); image counts are batches x batch size, so the last partial batch is counted as full.
- Runs R01-R05 were executed sequentially on one Apple MPS device; ledger cost is measured on the same machine.
