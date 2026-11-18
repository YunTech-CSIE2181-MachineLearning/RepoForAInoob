# Topic 10 experiment results

This directory contains the latest **27** supervised ResNet-18 strategy-comparison runs. The three dimensions are:

- **Category:** `carpet`, `bottle`, `transistor`
- **Training mode:** `scratch`, `frozen`, `finetune`
- **Seed:** `0`, `1`, `2`

Each run directory is named `{category}_{mode}_seed{seed}` and contains `history.csv` (per-epoch metrics) and `status.json` (configuration and selected best checkpoint). The local-only checkpoint files `best.pt` and `last.pt` are **not stored in GitHub**.

`summary.csv` is generated from all 27 run records by `uv run python scripts/verify_results.py`. `splits/` contains the three fixed CSVs created using Topic 10's two-stage stratified split (seed 42).

**Result definitions:** `best_val_auroc` is the AUROC of the epoch with **minimum validation loss**. It is not necessarily the maximum AUROC observed during training. All these values are **validation** results, not final held-out test results.

The previous experiment protocol is deliberately **not** included here; it was removed from repository tracking during cleanup. Training environment, full method mapping and rerun commands are documented in the root README and `docs/TOPIC10_CODE_MAP.md`.
