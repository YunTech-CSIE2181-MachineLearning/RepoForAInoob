"""Audit Train/Validation integrity without touching held-out images.

Run: uv run python scripts/audit_splits.py

Exact duplicate = same SHA-256 bytes under different paths (conclusive).
Near duplicate = small 64-bit difference hash distance (inspection hint only).
Neither changes any CSV or training checkpoint.
"""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("carpet", "bottle", "transistor")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path):
    """64-bit horizontal difference hash; useful only as a visual similarity hint."""
    with Image.open(path) as im:
        pixels = list(im.convert("L").resize(
            (9, 8), resample=Image.Resampling.LANCZOS
        ).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def audit(category, threshold, max_pairs):
    csv_path = ROOT / "splits" / f"{category}_split_seed42.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    paths = [r["path"] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{category}: duplicated file paths in CSV")

    counters = Counter(r["split"] for r in rows)
    if set(counters) != {"train", "val", "heldout"}:
        raise ValueError(f"{category}: invalid split values")
    for r in rows:
        if (r["defect_type"] == "good") != (r["label"] == "0"):
            raise ValueError(f"{category}: inconsistent label for {r['path']}")

    # Do not open any images belonging to held-out.
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    val_types = set(r["defect_type"] for r in val)
    all_types = set(r["defect_type"] for r in rows)
    if val_types != all_types:
        raise ValueError(f"{category}: missing validation types {all_types - val_types}")

    a, b = [], []
    for group, dest in ((train, a), (val, b)):
        for r in group:
            path = ROOT / r["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            dest.append((r, sha256_file(path), dhash(path)))

    exact = [(x[0]["path"], y[0]["path"]) for x in a for y in b if x[1] == y[1]]
    near = sorted(
        [(int((x[2] ^ y[2]).bit_count()), x[0]["path"], y[0]["path"])
         for x in a for y in b
         if x[1] != y[1] and (x[2] ^ y[2]).bit_count() <= threshold]
    )

    print(f"\n{category}: train={len(train)}, val={len(val)}, heldout={counters['heldout']} (not opened)")
    print(f"  Train/Val exact SHA-256 duplicate pairs: {len(exact)}")
    for p, q in exact[:max_pairs]:
        print(f"    EXACT {p} <-> {q}")
    print(f"  Train/Val near-similar dHash pairs (distance <= {threshold}): {len(near)}")
    for distance, p, q in near[:max_pairs]:
        print(f"    dHash={distance}: {p} <-> {q}")
    if near:
        print("  Note: dHash can flag normal industrial photos that only look similar.")
    return len(exact)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--near-threshold", type=int, default=4)
    p.add_argument("--max-pairs", type=int, default=8)
    args = p.parse_args()
    if not 0 <= args.near_threshold <= 64 or args.max_pairs < 0:
        p.error("invalid distance or pair limit")

    count = sum(audit(c, args.near_threshold, args.max_pairs) for c in CATEGORIES)
    print(f"\nTotal cross-split exact duplicate pairs: {count}")
    print("Held-out image contents were not read. Dataset splits were not changed.")
    if count:
        raise SystemExit("WARNING: exact train/val duplicates found; inspect before interpreting AUROC")


if __name__ == "__main__":
    main()
