from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "mvtec_ad"
SPLIT_DIR = ROOT / "splits"

# Fixed values from the submitted proposal.
CATEGORIES = ["carpet", "bottle", "transistor"]
SEED = 42
HELDOUT_RATIO = 0.5
VAL_RATIO = 0.2


def ensure_validation_coverage(train, val):
    """Cover every defect type in val without changing held-out or split sizes.

    Only move samples between train and val. A sample from an unrepresented
    defect type replaces a sample from an overrepresented type of the same
    binary class whenever possible. This keeps Good/Defect counts unchanged.
    """
    dev = pd.concat([train, val])
    train, val = train.copy(), val.copy()
    targets = sorted(dev["defect_type"].unique())

    for missing_type in targets:
        if (val["defect_type"] == missing_type).any():
            continue

        candidates = train[train["defect_type"] == missing_type].sort_values("path")
        if candidates.empty:
            raise ValueError(f"No training image available for {missing_type}")
        incoming = candidates.index[0]
        label = int(train.loc[incoming, "label"])

        # Do not eliminate any existing type from validation.
        counts = val["defect_type"].value_counts()
        donors = val[val["defect_type"].map(counts) > 1]
        same_label = donors[donors["label"] == label]
        if not same_label.empty:
            donors = same_label
        if donors.empty:
            raise ValueError("No safe validation donor; inspect split manually")

        # Choose the type most above its proportional expected validation count.
        expected = dev["defect_type"].value_counts() * (len(val) / len(dev))
        donor_type = sorted(
            donors["defect_type"].unique(),
            key=lambda t: (-(counts[t] - expected[t]), t),
        )[0]
        outgoing = donors[donors["defect_type"] == donor_type].sort_values("path").index[0]
        train = pd.concat([train.drop(index=incoming), val.loc[[outgoing]]])
        val = pd.concat([val.drop(index=outgoing), candidates.loc[[incoming]]])

    if set(val["defect_type"]) != set(targets):
        raise AssertionError("Incomplete validation defect-type coverage")
    return train, val


def prepare_category(category):
    test_dir = DATA_ROOT / category / "test"
    mask_dir = DATA_ROOT / category / "ground_truth"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {test_dir}")

    rows = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        for image_path in sorted(defect_dir.glob("*.png")):
            label = int(defect_type != "good")
            mask_path = ""
            if label:
                mask = mask_dir / defect_type / f"{image_path.stem}_mask.png"
                if not mask.is_file():
                    raise FileNotFoundError(f"Missing defect mask: {mask}")
                mask_path = mask.relative_to(ROOT).as_posix()
            rows.append({
                "category": category,
                "path": image_path.relative_to(ROOT).as_posix(),
                "defect_type": defect_type,
                "label": label,
                "mask": mask_path,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images for {category}")

    # Stage 1: retain the original 50/50 defect-type-stratified held-out split.
    dev, heldout = train_test_split(
        df, test_size=HELDOUT_RATIO,
        stratify=df["defect_type"], random_state=SEED,
    )

    # Stage 2: preserve the original split, correcting only missing types in val.
    train, val = train_test_split(
        dev, test_size=VAL_RATIO,
        stratify=dev["label"], random_state=SEED,
    )
    train, val = ensure_validation_coverage(train, val)

    df["split"] = ""
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[heldout.index, "split"] = "heldout"

    assert df["split"].ne("").all()
    assert df["path"].is_unique
    assert len(train) + len(val) + len(heldout) == len(df)

    output = SPLIT_DIR / f"{category}_split_seed{SEED}.csv"
    if output.exists():
        old = pd.read_csv(output).fillna("")
        new = df.fillna("")
        if not old.equals(new):
            raise RuntimeError(
                f"{output.name} differs from the fixed split. "
                "Will not overwrite. Check GitHub and local files."
            )
        print(f"{category}: fixed CSV verified")
    else:
        df.to_csv(output, index=False, encoding="utf-8")
        print(f"{category}: created {output.name}")

    print(f"  Total={len(df)} Train={len(train)} Val={len(val)} Held-out={len(heldout)}")
    print(pd.crosstab(df["defect_type"], df["split"]))
    print()


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        prepare_category(category)
    print("All three fixed splits verified.")


if __name__ == "__main__":
    main()
