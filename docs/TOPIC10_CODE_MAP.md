# Topic 10 notebook → project code map

The **source of section numbering** is `notebooks/Topic10_Transfer_Learning_MVTecAD.ipynb`, the original course notebook. Sections below identify where classmates can find corresponding concepts in this repository. The mapping is conceptual: some code is adapted to the command-line workflow and is **not** a verbatim copy.

| Topic 10 section | Course notebook topic | Project entry point / implementation |
| --- | --- | --- |
| **§0** | Environment check | `uv sync` and the CUDA check in `scripts/train_baseline.py:main()`; see README. |
| **§1** | Settings and seeds | Constants, `MODES` and `set_seed()` in `scripts/train_baseline.py`; `SPLIT_SEED` and `CATEGORIES` in `scripts/prepare_splits.py`. |
| **§2** | Locate MVTec AD | `DATA_ROOT` in `scripts/prepare_splits.py`; local `data/mvtec_ad` directory. No automatic download. |
| **§3** | Stratified train/val/held-out split | `build_split()` in `scripts/prepare_splits.py`; `load_split()` in `scripts/train_baseline.py`. |
| **§4** | Dataset, image cache and transforms | `cached()`, `DefectDataset`, `build_transform()` in `scripts/train_baseline.py`. |
| **§5** | CNN backbone and classification head | `WEIGHTS`, `build_model()`, `count_params()` in `scripts/train_baseline.py`. |
| **§6** | Training, evaluation and early stopping | `train()` and `keep_frozen_bn_in_eval()` in `scripts/train_baseline.py`. |
| **§7** | Scratch vs frozen vs full fine-tuning | `MODES` and `train()`; invoke `--mode all --seed <0|1|2>`. |
| **§8** | Other architectures | Demonstrated **only in the original notebook**; not included in current 27-run project. |
| **§9** | Augmentation and label distortion | Demonstrated **only in the original notebook**; not included in these no-augmentation runs. |
| **§10** | ROC, threshold, confusion matrix | Reserved for final evaluation; not implemented in the current training scripts. |
| **§11** | False positives/negatives and Grad-CAM | Reserved for final error analysis; refer to original notebook. |
| **§12** | Experiment records and reproducibility | `history.csv`, `status.json`, `.pt` checkpoints; `scripts/verify_results.py` aggregates committed records. |
| **§12b** | GitHub upload | Maintainers commit locally and push; no automatic push performed by training. |

## Exact run protocol (Topic 10 comparison)

| Mode | Pretrained backbone? | Train backbone? | Backbone LR | Head LR |
| --- | --- | --- | --- | --- |
| `scratch` | No | Yes | `1e-3` | `1e-3` |
| `frozen` | Yes | No | — | `1e-3` |
| `finetune` | Yes | Yes | `1e-4` | `1e-4` |

All 27 experiments use a ResNet-18 two-class head, AdamW with `weight_decay=0.05`, cosine learning-rate scheduling, balanced class-weighted **training** Cross Entropy, unweighted **validation** Cross Entropy, up to 30 epochs with validation-loss early stopping (patience 6), seed-specific training shuffling, and **no augmentation**. A model's `best.pt` is selected by **minimum validation loss**, not the highest epoch AUROC.

## Important adaptations and limitations

1. **Held-out is protected.** The teacher's `run_experiment()` demonstrates held-out scoring for each run. The project's scripts only load `train` and `val` during training; use held-out once at the end after model selection.
2. **CSV paths are relative.** The split membership follows the notebook's two-stage `train_test_split` procedure without additional validation type-coverage swapping. Relative paths enable cloning on other machines provided the dataset is at `data/mvtec_ad`.
3. **Resuming runs is supported.** `last.pt` includes optimizer, scheduler and RNG states. Existing completed runs are skipped. Model checkpoints are deliberately untracked to avoid large repository uploads.
4. **This is not the Week 12 study.** The required optimizer/learning-rate/weight-decay grid and augmentation experiments have not been included in these 27 results. Do not claim otherwise.
5. **Validation AUROC is not a held-out score.** Validation has very few examples; values of `1.0` should not be presented as generalization guarantees.
6. **Classroom execution is unverified.** The classroom GPU and dataset path require local testing; no working ST301 run is claimed here.

## How to inspect one experiment

For example, `outputs_topic10/transistor_frozen_seed1/` contains:

- `history.csv` — epoch-by-epoch training loss, validation loss and validation AUROC;
- `status.json` — chosen epoch (`best_epoch`), its validation AUROC (`best_val_auroc`), its validation loss (`best_val_loss`), full mode/seed settings, and completion state;
- `best.pt` and `last.pt` — available **only on the training computer** unless trained again.

`uv run python scripts/verify_results.py` verifies 27 complete records and writes `outputs_topic10/summary.csv` in a format useful for tables and plots. It uses only Python's standard library; no GPU or MVTec images are required.
