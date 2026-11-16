"""Reproduce Topic 10's training protocol for 3 categories x 3 modes x 3 seeds.

Based on notebooks/Topic10_Transfer_Learning_MVTecAD.ipynb:
- AdamW + cosine LR, class-weighted training loss, validation-loss early stopping;
- mode-specific backbone/head learning rates, independent shuffled DataLoader seed;
- cached bilinear 256px images, ResNet-18, no augmentation.
Adaptations: Track A held-out isolation, resumable checkpoints, reproducible logs,
and a SEPARATE outputs_topic10/ directory so old results are not overwritten.

Prepare the new splits before training:
    uv run python scripts/prepare_splits.py

Run 27 models:
    uv run python scripts/train_baseline.py --mode all --seed 0
    uv run python scripts/train_baseline.py --mode all --seed 1
    uv run python scripts/train_baseline.py --mode all --seed 2
"""
import argparse
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("carpet", "bottle", "transistor")
SPLIT_SEED = 42
CACHE_SIZE = 256
PATIENCE = 6
WEIGHT_DECAY = 0.05
PROTOCOL = "topic10-adamw-cosine-weighted-loss-earlystop-v1"
# Same ImageNet normalization and baseline (no-augmentation) transform as Topic 10.
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Keep only "none" here: augmentation experiments have their own planned settings.
AUGMENTATIONS = {"none": ([], [])}

def build_transform(aug="none", train=True, size=IMG_SIZE):
    pil_ops, tensor_ops = [], []
    if train:
        for name in aug.split("+"):
            p, t = AUGMENTATIONS[name]
            pil_ops += p; tensor_ops += t
    return T.Compose([T.Resize((size, size)), *pil_ops,
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD), *tensor_ops])

# Exact mode-specific learning rates from Topic 10's transfer strategy comparison.
# This intentionally changes the previous fixed-LR protocol; old outputs are kept.
MODES = {
    #                pretrained, train_backbone, backbone_lr, head_lr
    "scratch":        (False, True,  1e-3, 1e-3),
    "frozen":         (True,  False, 0.0,  1e-3),
    "finetune":       (True,  True,  1e-4, 1e-4),
    "discriminative": (True,  True,  1e-4, 1e-3),
}

WEIGHTS = {
    "resnet18":          models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50":          models.ResNet50_Weights.IMAGENET1K_V2,
    "mobilenet_v3_large":models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    "efficientnet_b0":   models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "vgg16":             models.VGG16_Weights.IMAGENET1K_V1,
}

def build_model(arch="resnet18", pretrained=True, num_classes=2):
    # 回傳 (model, head)：head 是新換上去、要從零學的分類層所在模組
    weights = WEIGHTS[arch] if pretrained else None
    m = getattr(models, arch)(weights=weights)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    else:  # mobilenet / efficientnet / vgg 的分類器都在 m.classifier，最後一個是 Linear
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        head = m.classifier
    return m, head

def count_params(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)

def keep_frozen_bn_in_eval(model):
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()


# Only training and validation images are cached. No held-out pixels are touched.
_CACHE = {}


def cached(path):
    if path not in _CACHE:
        with Image.open(ROOT / path) as im:
            img = im.convert("RGB")
            _CACHE[path] = img.resize(
                (CACHE_SIZE, CACHE_SIZE), Image.Resampling.BILINEAR
            )
    return _CACHE[path]


class DefectDataset(Dataset):
    """Equivalent to Topic 10's dataset, using portable CSV dictionaries."""
    def __init__(self, rows, transform):
        self.rows = rows
        self.tf = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(cached(r["path"])), int(r["label"]), i


def load_split(category):
    csv_path = ROOT / "outputs_topic10" / "splits" / f"{category}_split_seed{SPLIT_SEED}.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate data in {csv_path}")
    if set(r["split"] for r in rows) != {"train", "val", "heldout"}:
        raise ValueError(f"Invalid split labels in {csv_path}")
    result = {}
    for split in ("train", "val"):
        subset = [r for r in rows if r["split"] == split]
        if set(int(r["label"]) for r in subset) != {0, 1}:
            raise ValueError(f"{category}: {split} must have both classes")
        missing = [r["path"] for r in subset if not (ROOT / r["path"]).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        result[split] = subset
    # Intentionally never open held-out image or mask paths.
    return result


def set_seed(seed: int):
    # 固定所有亂數來源：Python / NumPy / PyTorch(CPU+GPU)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def train(category, mode, seed, epochs, batch_size, device, resume):
    set_seed(seed)
    groups = load_split(category)
    run_id = f"{category}_{mode}_seed{seed}"
    out = ROOT / "outputs_topic10" / run_id
    out.mkdir(parents=True, exist_ok=True)
    progress_file = out / "last.pt"
    best_file = out / "best.pt"
    status_file = out / "status.json"
    log_file = out / "history.csv"

    # Match Topic 10 (cell 24) exactly for mode-specific learning rates.
    pretrained, train_backbone, lr_bb, lr_head = MODES[mode]
    model, head = build_model("resnet18", pretrained)
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = train_backbone or id(p) in head_ids
    model.to(device)

    param_groups = [{"params": list(head.parameters()), "lr": lr_head}]
    bb = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    if bb:
        param_groups.append({"params": bb, "lr": lr_bb})
    opt = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    trainable = count_params(model, trainable_only=True)
    total = count_params(model)
    if mode == "frozen":
        assert trainable == 1026
    print(f"{run_id}: trainable={trainable:,}/{total:,}; backbone_lr={lr_bb}; head_lr={lr_head}", flush=True)

    counts = np.bincount([int(r["label"]) for r in groups["train"]], minlength=2)
    class_w = torch.tensor(
        len(groups["train"]) / (2 * np.maximum(counts, 1)),
        dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # Topic 10 gives each training loader an independent, seed-specific generator.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        DefectDataset(groups["train"], build_transform("none", train=True)),
        batch_size=batch_size, shuffle=True, generator=g, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        DefectDataset(groups["val"], build_transform("none", train=False)),
        batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
    )

    # Both datasets are now small in-memory caches, just like the course notebook.
    for subset in ("train", "val"):
        for row in groups[subset]:
            cached(row["path"])


    # Historical results used a different protocol. Never reuse their checkpoints.
    # Store scheduler, DataLoader generator and RNG states for exact resumption.
    start, best_loss, best_epoch, bad = 1, float("inf"), 0, 0
    best_auc = float("nan")
    history = []
    finished = False
    if resume and progress_file.is_file():
        state = torch.load(progress_file, map_location=device, weights_only=False)
        expected = (PROTOCOL, category, mode, seed, epochs, batch_size)
        actual = tuple(state.get(k) for k in
                       ("protocol", "category", "mode", "seed", "epochs", "batch_size"))
        if expected != actual:
            raise ValueError(f"Incompatible protocol/checkpoint: {progress_file}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        g.set_state(state["rng_loader"])
        start = state["epoch"] + 1
        best_loss, best_auc, best_epoch = (
            state["best_val_loss"], state["best_val_auroc"], state["best_epoch"]
        )
        bad, history = state["bad_epochs"], state["history"]
        finished = state["finished"]
        torch.set_rng_state(state["rng_torch"].cpu())
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        print(f"RESUME {run_id} from epoch {start}", flush=True)
    elif not resume and progress_file.exists():
        raise RuntimeError(
            f"{progress_file} exists. --no-resume cannot overwrite an existing run. "
            "Move the run directory elsewhere first."
        )

    if finished or start > epochs:
        print(f"COMPLETE {run_id}: best val loss={best_loss:.4f}, "
              f"val AUROC={best_auc:.4f}, epoch={best_epoch}", flush=True)
        return

    print(f"START {run_id} train={len(groups['train'])}, val={len(groups['val'])}", flush=True)
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        if not train_backbone:
            keep_frozen_bn_in_eval(model)

        loss_sum = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(y)
        sched.step()

        # Match teacher's predict(): VALIDATION LOSS IS UNWEIGHTED.
        # Training loss above is class-weighted, unlike validation loss.
        model.eval()
        ys, scores, val_total = [], [], 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits = model(x.to(device))
                val_total += F.cross_entropy(
                    logits, y.to(device), reduction="sum"
                ).item()
                scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
                ys.extend(y.tolist())
        val_loss = val_total / len(val_loader.dataset)
        val_auc = float(roc_auc_score(ys, scores))

        row = dict(
            epoch=epoch, train_loss=loss_sum / len(train_loader.dataset),
            val_loss=val_loss, val_auroc=val_auc,
            seconds=round(time.time() - t0, 2)
        )
        history.append(row)
        # Teacher's checkpoint selection and stopping: best val loss,
        # improvement > 1e-4, patience 6 (rather than AUROC-selected epoch).
        if val_loss < best_loss - 1e-4:
            best_loss, best_auc, best_epoch, bad = val_loss, val_auc, epoch, 0
            best_state = {
                "protocol": PROTOCOL, "model": model.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "val_auroc": val_auc,
                "category": category, "mode": mode, "seed": seed,
            }
            tmp_best = out / "best.tmp"
            torch.save(best_state, tmp_best)
            os.replace(tmp_best, best_file)
        else:
            bad += 1

        with log_file.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        finished = bad >= PATIENCE or epoch == epochs
        state = dict(
            protocol=PROTOCOL, category=category, mode=mode, seed=seed,
            epochs=epochs, batch_size=batch_size, epoch=epoch,
            best_val_loss=best_loss, best_val_auroc=best_auc,
            best_epoch=best_epoch, bad_epochs=bad, finished=finished,
            history=history, model=model.state_dict(),
            optimizer=opt.state_dict(), scheduler=sched.state_dict(),
            rng_loader=g.get_state(), rng_torch=torch.get_rng_state(),
            rng_numpy=np.random.get_state(), rng_python=random.getstate(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        )
        tmp_ckpt = out / "last.tmp"
        torch.save(state, tmp_ckpt)
        os.replace(tmp_ckpt, progress_file)
        write_json_atomic(status_file, dict(
            protocol=PROTOCOL, category=category, mode=mode, seed=seed,
            epoch=epoch, epochs=epochs, status="complete" if finished else "running",
            best_val_loss=best_loss, best_val_auroc=best_auc,
            best_epoch=best_epoch, bad_epochs=bad, patience=PATIENCE,
            train_count=len(groups["train"]), val_count=len(groups["val"]),
            optimizer="AdamW", weight_decay=WEIGHT_DECAY,
            backbone_lr=lr_bb, head_lr=lr_head,
            lr_schedule="CosineAnnealingLR",
            train_loss="weighted_cross_entropy",
            val_loss="unweighted_cross_entropy",
            selection="lowest_validation_loss",
            image_size=IMG_SIZE, cache_size=CACHE_SIZE,
            batch_size=batch_size, split_seed=SPLIT_SEED,
            augmentation="none", trainable_parameters=trainable,
            total_parameters=total,
            git_commit=git_hash(), device=str(device),
            elapsed_epoch_seconds=row["seconds"]
        ))
        print(f"{run_id} epoch {epoch:02d}/{epochs} "
              f"train loss {row['train_loss']:.4f} "
              f"val loss {val_loss:.4f} val AUROC {val_auc:.4f} "
              f"best loss {best_loss:.4f} bad {bad}/{PATIENCE}", flush=True)
        if finished:
            break
    print(f"DONE {run_id}: best validation loss={best_loss:.4f}; "
          f"AUROC at best epoch={best_auc:.4f} (epoch {best_epoch})", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",
                   choices=("scratch", "frozen", "finetune", "pretrained", "both", "all"),
                   default="scratch")
    p.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    if args.seed < 0 or args.epochs < 1 or args.batch_size < 1:
        p.error("seed >= 0, epochs and batch-size >= 1 required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available; refusing slow CPU training")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    if args.mode == "both":
        modes = ["scratch", "finetune"]
    elif args.mode == "all":
        modes = ["scratch", "frozen", "finetune"]
    else:
        modes = ["finetune" if args.mode == "pretrained" else args.mode]
    for mode in modes:
        for category in args.categories:
            train(category, mode, args.seed, args.epochs, args.batch_size,
                  device, not args.no_resume)


if __name__ == "__main__":
    main()
