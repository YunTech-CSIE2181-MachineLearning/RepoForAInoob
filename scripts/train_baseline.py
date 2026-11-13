"""Week 11 Track A baselines: scratch and ImageNet-pretrained ResNet-18.

No augmentation; NEVER reads held-out images. Invoke from repository root:
    uv run python scripts/train_baseline.py --mode scratch
    uv run python scripts/train_baseline.py --mode pretrained
    uv run python scripts/train_baseline.py --mode both
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("carpet", "bottle", "transistor")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageSet(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(ROOT / r["path"]) as im:
            x = TF(im.convert("RGB"))
        return x, int(r["label"])


def load_split(category):
    csv_path = ROOT / "splits" / f"{category}_split_seed42.csv"
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


def seed_all(seed):
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


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    scores, labels, total_loss = [], [], 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * len(targets)
        scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(targets.cpu().tolist())
    return total_loss / len(loader.dataset), float(roc_auc_score(labels, scores))


def train(category, mode, seed, epochs, batch_size, device, resume):
    seed_all(seed)
    groups = load_split(category)
    run_id = f"{category}_{mode}_seed{seed}"
    out = ROOT / "outputs_baseline" / run_id
    out.mkdir(parents=True, exist_ok=True)
    progress_file = out / "last.pt"
    best_file = out / "best.pt"
    status_file = out / "status.json"
    log_file = out / "history.csv"

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if mode == "pretrained" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(ImageSet(groups["train"]), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(ImageSet(groups["val"]), batch_size=batch_size,
                            shuffle=False, num_workers=0)

    start, best_auc, best_epoch = 1, -1.0, 0
    history = []
    if resume and progress_file.is_file():
        state = torch.load(progress_file, map_location=device, weights_only=False)
        if (state["category"], state["mode"], state["seed"],
                state["epochs"], state["batch_size"]) != (
                category, mode, seed, epochs, batch_size):
            raise ValueError(f"Incompatible checkpoint: {progress_file}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        best_auc = state["best_val_auroc"]
        best_epoch = state["best_epoch"]
        history = state["history"]
        torch.set_rng_state(state["rng_torch"].cpu())
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        print(f"RESUME {run_id} at epoch {start}", flush=True)

    if start > epochs:
        print(f"COMPLETE {run_id}, best val AUROC={best_auc:.4f} at epoch {best_epoch}")
        return

    print(f"START {run_id} train={len(groups['train'])}, val={len(groups['val'])}", flush=True)
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        row = dict(epoch=epoch, train_loss=loss_sum / len(train_loader.dataset),
                   val_loss=val_loss, val_auroc=val_auc, seconds=round(time.time() - t0, 2))
        history.append(row)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auroc": val_auc, "category": category, "mode": mode}, best_file)

        with log_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        # Checkpoint after each epoch. Resume restores RNG and optimizer state.
        state = dict(category=category, mode=mode, seed=seed, epochs=epochs,
                     batch_size=batch_size, epoch=epoch, best_val_auroc=best_auc,
                     best_epoch=best_epoch, history=history, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), rng_torch=torch.get_rng_state(),
                     rng_numpy=np.random.get_state(), rng_python=random.getstate(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        temp_ckpt = out / "last.tmp"
        torch.save(state, temp_ckpt)
        os.replace(temp_ckpt, progress_file)
        write_json_atomic(status_file, dict(category=category, mode=mode, seed=seed,
            epoch=epoch, epochs=epochs, best_val_auroc=best_auc, best_epoch=best_epoch,
            status="complete" if epoch == epochs else "running",
            train_count=len(groups["train"]), val_count=len(groups["val"]),
            git_commit=git_hash(), device=str(device), elapsed_epoch_seconds=row["seconds"],
            optimizer="Adam", lr=1e-4, image_size=224, batch_size=batch_size,
            split_seed=42, augmentation="none"))
        print(f"{run_id} epoch {epoch:02d}/{epochs} | "
              f"train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val AUROC {val_auc:.4f} | best {best_auc:.4f} | {row['seconds']}s",
              flush=True)
    print(f"DONE {run_id}: best validation AUROC={best_auc:.4f} (epoch {best_epoch})", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("scratch", "pretrained", "both"), default="scratch")
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
    modes = ["scratch", "finetune"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for category in args.categories:
            train(category, mode, args.seed, args.epochs, args.batch_size,
                  device, not args.no_resume)


if __name__ == "__main__":
    main()
