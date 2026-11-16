"""Week 11 Track A: Scratch, Frozen Features, Full Fine-tuning ResNet-18.

No augmentation; NEVER opens held-out images. Run from repository root:
    uv run python scripts/train_baseline.py --mode frozen --seed 0
    uv run python scripts/train_baseline.py --mode frozen --seed 1
    uv run python scripts/train_baseline.py --mode frozen --seed 2

Modes: scratch, frozen, finetune, both (scratch + finetune), all.
'pretrained' is a backward-compatible alias for 'finetune'.
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
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("carpet", "bottle", "transistor")
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

# Track A transfer strategies, following Topic 10's mode interpretation.
# LR remains fixed at 1e-4 to preserve the existing experiment configuration.
MODES = {
    "scratch": (False, True),
    "frozen": (True, False),
    "finetune": (True, True),
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


class DefectDataset(Dataset):
    """Adapted from Topic 10: use fixed CSV rows, not its cached DataFrame.

    Return only (image, label) to preserve the existing training/validation
    interfaces and avoid accidentally loading held-out samples.
    """
    def __init__(self, rows, transform):
        self.rows = rows
        self.tf = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(ROOT / r["path"]) as im:
            x = self.tf(im.convert("RGB"))
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
    set_seed(seed)
    groups = load_split(category)
    run_id = f"{category}_{mode}_seed{seed}"
    out = ROOT / "outputs_baseline" / run_id
    out.mkdir(parents=True, exist_ok=True)
    progress_file = out / "last.pt"
    best_file = out / "best.pt"
    status_file = out / "status.json"
    log_file = out / "history.csv"

    # Same transfer-learning setup as Topic 10; uses the original ResNet-18.
    pretrained, train_backbone = MODES[mode]
    model, head = build_model("resnet18", pretrained)
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = train_backbone or id(p) in head_ids
    if mode == "frozen":
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        assert trainable_names == {"fc.weight", "fc.bias"}, trainable_names
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = count_params(model, trainable_only=True)
    n_total = count_params(model)
    if mode == "frozen":
        assert n_trainable == 1026, f"Unexpected frozen head size: {n_trainable}"
    print(f"{run_id}: trainable={n_trainable:,} / total={n_total:,} parameters", flush=True)
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Keep the ORIGINAL DataLoader shuffling/RNG behavior for comparability.
    train_loader = DataLoader(
        DefectDataset(groups["train"], build_transform(train=True)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        DefectDataset(groups["val"], build_transform(train=False)),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    start, best_auc, best_epoch = 1, -1.0, 0
    history = []
    if resume and progress_file.is_file():
        state = torch.load(progress_file, map_location=device, weights_only=False)
        # Existing fine-tuning checkpoints were saved as mode='pretrained'.
        saved_mode = "finetune" if state["mode"] == "pretrained" else state["mode"]
        if (state["category"], saved_mode, state["seed"],
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
        if mode == "frozen":
            # Same BatchNorm freeze helper as Topic 10.
            keep_frozen_bn_in_eval(model)
            assert all(not layer.training for layer in model.modules()
                       if isinstance(layer, nn.modules.batchnorm._BatchNorm))
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
            split_seed=42, augmentation="none",
            trainable_parameters=n_trainable, total_parameters=n_total,
            backbone_frozen=(mode == "frozen")))
        print(f"{run_id} epoch {epoch:02d}/{epochs} | "
              f"train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val AUROC {val_auc:.4f} | best {best_auc:.4f} | {row['seconds']}s",
              flush=True)
    print(f"DONE {run_id}: best validation AUROC={best_auc:.4f} (epoch {best_epoch})", flush=True)


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
