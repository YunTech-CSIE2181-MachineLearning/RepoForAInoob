# Week 11 baseline check-in (2026-11-19)

The course's revised instructions define **baseline** as ResNet-18 **trained
from scratch with no augmentation** (slide: Required pipeline). The submitted
proposal also specifies an **ImageNet-pretrained ResNet-18** with Adam,
learning rate 0.0001, batch size 16, 30 epochs and validation-AUROC checkpoint
selection. This script offers BOTH variants, and labels them separately.

## Start on the RTX 4060 PC

Open PowerShell at the repository root and run:

```powershell
git pull flash main
uv sync
uv run python scripts/train_baseline.py --mode scratch
uv run python scripts/train_baseline.py --mode pretrained
```

Or run both sequentially: `uv run python scripts/train_baseline.py --mode both`.
To do a very small **smoke test** before the full experiment, use:

```powershell
uv run python scripts/train_baseline.py --mode scratch --categories carpet --epochs 1
```

**Important:** that test creates a one-epoch checkpoint under the same run
name. Before starting a full 30-epoch run, remove ONLY the
`outputs_baseline/carpet_scratch_seed0` test directory, or run with
`--no-resume` (the latter starts over and overwrites the existing run).
Do not interpret the smoke-test score as a full baseline result.

The fixed split CSVs in `splits/` must already exist. Only their `train`
and `val` rows are loaded. **Held-out images are not opened or scored.**
ImageNet normalization and resize to 224×224 are the only transforms.
Each category is trained independently; each run saves `history.csv`,
`status.json`, `best.pt`, and `last.pt` inside
`outputs_baseline/<category>_<mode>_seed0/`.

Every epoch creates a resumable checkpoint. Restarting with the same command
resumes a partial run or skips a completed one. Keep the computer awake,
and close the process cleanly before changing the code or parameters.
The checkpoint includes optimizer and random generator state but exact
reproducibility may still depend on CUDA implementation/environment.

For a Week 11 check-in show **validation AUROC** for all three categories,
each separately labeled `scratch` or `pretrained`. Final held-out scores
must be computed only after experimental choices are finalized. Subsequent
final-report experiments should include 3 training seeds; the default Week 11
check-in uses seed 0 as an initial result.

**Git note:** keep checkpoint weights local. Do not commit large model files.
