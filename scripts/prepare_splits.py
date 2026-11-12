

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 專案根目錄
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "mvtec_ad"
SPLIT_DIR = ROOT / "splits"

# Proposal 規定
CATEGORIES = ["carpet", "bottle", "transistor"]
SEED = 42
HELDOUT_RATIO = 0.5
VAL_RATIO = 0.2


def prepare_category(category):
    test_dir = DATA_ROOT / category / "test"
    mask_dir = DATA_ROOT / category / "ground_truth"

    if not test_dir.is_dir():
        raise FileNotFoundError(f"找不到資料夾: {test_dir}")

    rows = []

    # 讀取每一種 defect type
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name

        for image_path in sorted(defect_dir.glob("*.png")):
            label = 0 if defect_type == "good" else 1

            mask_path = ""
            if label == 1:
                mask = (
                    mask_dir
                    / defect_type
                    / f"{image_path.stem}_mask.png"
                )

                if not mask.is_file():
                    raise FileNotFoundError(
                        f"缺少瑕疵標註: {mask}"
                    )

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
        raise ValueError(f"{category} 沒有找到任何圖片")

    # 第一階段：依 defect type 分層，50% dev / 50% heldout
    dev, heldout = train_test_split(
        df,
        test_size=HELDOUT_RATIO,
        stratify=df["defect_type"],
        random_state=SEED,
    )

    # 第二階段：從 dev 取 20% validation
    # 依 Good / Defect 分層
    train, val = train_test_split(
        dev,
        test_size=VAL_RATIO,
        stratify=dev["label"],
        random_state=SEED,
    )

    df["split"] = ""
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[heldout.index, "split"] = "heldout"

    # 基本驗證
    assert df["split"].ne("").all()
    assert df["path"].is_unique
    assert len(train) + len(val) + len(heldout) == len(df)

    # 如果已經存在切分檔，禁止無意間覆蓋
    output = SPLIT_DIR / f"{category}_split_seed{SEED}.csv"

    if output.exists():
        old = pd.read_csv(output).fillna("")
        new = df.fillna("")
        if not old.equals(new):
            raise RuntimeError(
                f"{output.name} 已存在且內容不同，"
                "為保護固定切分，不會覆蓋。"
            )
        print(f"{category}: 切分檔已存在，內容一致")
    else:
        df.to_csv(output, index=False, encoding="utf-8")
        print(f"{category}: 已建立 {output.name}")

    print(
        f"  Total={len(df)}, "
        f"Train={len(train)}, "
        f"Val={len(val)}, "
        f"Held-out={len(heldout)}"
    )

    print(pd.crosstab(df["defect_type"], df["split"]))
    print()


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        prepare_category(category)

    print("三個類別的資料切分檢查完成！")


if __name__ == "__main__":
    main()
