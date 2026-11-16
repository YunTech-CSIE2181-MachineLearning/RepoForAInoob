"""Create the Track A split using the *same two-stage procedure* as Topic 10.

The old 'splits/' files (and outputs_baseline/) are deliberately untouched.
This rerun writes to outputs_topic10/splits/, as in the course notebook.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "mvtec_ad"
OUT_DIR = ROOT / "outputs_topic10"
SPLIT_DIR = OUT_DIR / "splits"

CATEGORIES = ["carpet", "bottle", "transistor"]
SPLIT_SEED = 42


def build_split(root, category, seed=SPLIT_SEED, heldout_frac=0.5, val_frac=0.2):
    """Match Topic 10 cell 9: 50/50 stratified by defect type, then 20% val by label.

    Relative paths are used for portable CSVs; this does not change membership.
    No extra validation-coverage swapping (unlike our older splitter).
    """
    csv = SPLIT_DIR / f"{category}_split_seed{seed}.csv"
    rows = []
    for dt_dir in sorted((Path(root) / category / "test").iterdir()):
        if not dt_dir.is_dir():
            continue
        for f in sorted(dt_dir.glob("*.png")):
            m = Path(root) / category / "ground_truth" / dt_dir.name / f"{f.stem}_mask.png"
            rows.append(dict(
                category=category,
                path=f.relative_to(ROOT).as_posix(),
                defect_type=dt_dir.name,
                label=int(dt_dir.name != "good"),
                mask=m.relative_to(ROOT).as_posix() if m.exists() else "",
            ))
    df = pd.DataFrame(rows)
    if df.empty or not df["path"].is_unique:
        raise ValueError(f"Missing or duplicate data for {category}")

    # These three lines intentionally mirror Topic 10's build_split().
    dev, held = train_test_split(
        df, test_size=heldout_frac, stratify=df["defect_type"], random_state=seed
    )
    tr, va = train_test_split(
        dev, test_size=val_frac, stratify=dev["label"], random_state=seed
    )
    df["split"] = ""
    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[held.index, "split"] = "heldout"

    assert df["split"].ne("").all()
    assert len(tr) + len(va) + len(held) == len(df)
    if csv.exists():
        existing = pd.read_csv(csv).fillna("")
        if not existing.equals(df.fillna("")):
            raise RuntimeError(
                f"{csv} does not match the reproducible Topic 10 split. "
                "Refusing to overwrite; inspect the existing CSV first."
            )
        print(f"{category}: Topic 10 split verified")
    else:
        df.to_csv(csv, index=False)
        print(f"{category}: created {csv.relative_to(ROOT)}")
    print(f"  Total={len(df)} Train={len(tr)} Val={len(va)} Held-out={len(held)}")
    print(pd.crosstab(df["defect_type"], df["split"]))
    print()
    return df


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        if not (DATA_ROOT / category / "test").is_dir():
            raise FileNotFoundError(f"MVTec AD missing: {DATA_ROOT / category / 'test'}")
        build_split(DATA_ROOT, category)
    print("All three Topic 10 splits are ready (old splits/ preserved).")


if __name__ == "__main__":
    main()
