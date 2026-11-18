# Industrial Defect Classification on MVTec AD

**CISE2181 Machine Learning · Fall 2026 · Track A**  
**Team:** AInoob  
**Categories:** `carpet` (texture), `bottle` and `transistor` (objects)

This repository contains a reproducible, ResNet-18-based comparison of **training from scratch**, **frozen ImageNet features**, and **full fine-tuning** on a small supervised MVTec AD dataset. The implementation follows the course's **Topic 10** transfer-learning notebook, with small project-specific changes for repeatability and safe held-out evaluation.

> **Current scope:** 27 completed experiments (3 categories × 3 strategies × seeds 0, 1, and 2), using **validation** results. This is the Week 11 strategy comparison, **not** the final held-out evaluation or the Week 12 hyperparameter/augmentation study. Classroom (ST301) compatibility has **not yet been verified**.

## Find things quickly

| Path | Purpose |
| --- | --- |
| [`notebooks/Topic10_Transfer_Learning_MVTecAD.ipynb`](notebooks/Topic10_Transfer_Learning_MVTecAD.ipynb) | **Original course notebook** for Topic 10; kept unchanged as a reference. |
| [`scripts/prepare_splits.py`](scripts/prepare_splits.py) | Create/verify fixed train, validation, and held-out CSVs (Topic 10 **§3**). |
| [`scripts/train_baseline.py`](scripts/train_baseline.py) | Train **all three** strategies; the historical filename `train_baseline` does not mean scratch-only (Topic 10 **§4–§7, §12**). |
| [`scripts/verify_results.py`](scripts/verify_results.py) | Check the 27 result records and generate a compact summary without GPU or image data. |
| [`outputs_topic10/splits/`](outputs_topic10/splits/) | Three fixed split CSVs, split seed 42. |
| [`outputs_topic10/`](outputs_topic10/) | The **latest 27** run folders, with `history.csv` and `status.json`, plus summary files. |
| [`docs/TOPIC10_CODE_MAP.md`](docs/TOPIC10_CODE_MAP.md) | Detailed function-to-notebook section mapping and important implementation differences. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python dependencies and locked environment. |
| `data/mvtec_ad/` | **Local-only** MVTec AD images, intentionally excluded from Git. |
| [`environment_check/`](environment_check/) | One-command Windows setup, CUDA compatibility check, and ResNet-18 smoke training. |

## Requirements and data

- Python compatible with `pyproject.toml`, [`uv`](https://docs.astral.sh/uv/) and PyTorch/CUDA for training.
- MVTec AD folders for `carpet`, `bottle`, and `transistor`, located at **`data/mvtec_ad/` relative to this repository**. The CSVs store those relative paths.
- The script currently **requires a CUDA GPU** for training. The results verifier runs without one.

For a fresh checkout on Windows PowerShell:

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python scripts/verify_results.py
```

To provide the dataset, place the category folders at `data/mvtec_ad/`. If the classroom already provides the files elsewhere, you can create a **local directory junction** after confirming its actual path (example only):

```powershell
New-Item -ItemType Directory -Force data | Out-Null
New-Item -ItemType Junction -Path 'data\mvtec_ad' -Target 'D:\datasets\mvtec_ad'
```

The classroom dataset path, GPU, and training compatibility must be checked on that machine; this repository does not assume the classroom test has passed.

## Recreate the split and training runs

From the repository root:

```powershell
# §3: Creates or verifies outputs_topic10/splits/*.csv
uv run python scripts/prepare_splits.py

# §§4–7: Each command runs the 3 strategies over all 3 categories.
uv run python scripts/train_baseline.py --mode all --seed 0
uv run python scripts/train_baseline.py --mode all --seed 1
uv run python scripts/train_baseline.py --mode all --seed 2

# §12: Validate all 27 records and refresh outputs_topic10/summary.csv
uv run python scripts/verify_results.py
```

`--mode` also accepts `scratch`, `frozen`, and `finetune` individually. Each run has a unique name, such as `carpet_finetune_seed2`. Completed runs are recognized by compatible checkpoints and will be skipped when rerun. The experiment protocol uses ResNet-18, 224-pixel input, a 256-pixel image cache, AdamW (weight decay 0.05), cosine learning-rate scheduling, class-weighted training loss, at most 30 epochs, and validation-loss early stopping (patience 6). Mode-specific learning rates match the course's Topic 10 notebook.

**Important:** The scripts intentionally read only **train and validation** images while fitting and selecting checkpoints. The **held-out** half is reserved for the final project evaluation; do not tune with it. The original course notebook contains demonstrations that evaluate held-out data inside its experimental loop—those cells should not be used for model selection in this project.

## Interpreting output files

A completed run directory contains:

```text
outputs_topic10/carpet_finetune_seed2/
  history.csv     # One row per trained epoch: train loss, validation loss/AUROC
  status.json     # Protocol, hyperparameters, best validation epoch and score
  best.pt         # Local-only checkpoint selected by minimum validation loss
  last.pt         # Local-only resumable training state
```

The `.pt` checkpoints, MVTec images, and environment directories are ignored by Git. To recreate the weights, you must run training locally. **An AUROC of 1.0 on this very small validation set does not imply perfect held-out or production performance.**

See [`outputs_topic10/README.md`](outputs_topic10/README.md) for the result layout and [`docs/TOPIC10_CODE_MAP.md`](docs/TOPIC10_CODE_MAP.md) for the connection between these scripts and the teacher's notebook.

## Attribution and reproducibility

The original Topic 10 notebook is supplied by the course and remains in `notebooks/`. This project uses and adapts its training methods; project-specific scripts and repository documentation have been developed with **AI assistance**. The data are distributed separately under their applicable MVTec AD terms. No original image data or trained `.pt` weights are committed.
