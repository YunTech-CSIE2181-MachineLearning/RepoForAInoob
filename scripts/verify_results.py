"""Verify the 27 Topic 10 runs (no dataset/GPU required) and write a summary CSV.

Topic 10 §12: reproducibility and experiment records.
Run from any directory: uv run python scripts/verify_results.py
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs_topic10"
CATEGORIES = ("carpet", "bottle", "transistor")
MODES = ("scratch", "frozen", "finetune")
SEEDS = (0, 1, 2)
EXPECTED_PROTOCOL = "topic10-adamw-cosine-weighted-loss-earlystop-v1"


def verify_results(write_summary=True):
    errors = []
    runs = []
    expected = {f"{c}_{m}_seed{s}" for c in CATEGORIES for m in MODES for s in SEEDS}
    present = {p.name for p in RESULTS.iterdir() if p.is_dir() and p.name != "splits"} if RESULTS.is_dir() else set()
    if expected - present:
        errors.append(f"Missing run folders: {', '.join(sorted(expected - present))}")
    if present - expected:
        errors.append(f"Unexpected run folders: {', '.join(sorted(present - expected))}")

    for category in CATEGORIES:
        split = RESULTS / "splits" / f"{category}_split_seed42.csv"
        if not split.is_file():
            errors.append(f"Missing split CSV: {split.relative_to(ROOT)}")
        for mode in MODES:
            for seed in SEEDS:
                name = f"{category}_{mode}_seed{seed}"
                folder = RESULTS / name
                status_path, history_path = folder / "status.json", folder / "history.csv"
                if not status_path.is_file() or not history_path.is_file():
                    errors.append(f"{name}: missing history.csv or status.json")
                    continue
                try:
                    state = json.loads(status_path.read_text(encoding="utf-8-sig"))
                    with history_path.open(encoding="utf-8-sig", newline="") as f:
                        history = list(csv.DictReader(f))
                    if state.get("status") != "complete":
                        raise ValueError("status is not complete")
                    if state.get("protocol") != EXPECTED_PROTOCOL:
                        raise ValueError("unexpected protocol")
                    if (state.get("category"), state.get("mode"), state.get("seed")) != (category, mode, seed):
                        raise ValueError("category/mode/seed metadata mismatch")
                    if len(history) != int(state["epoch"]):
                        raise ValueError("epoch count does not match history rows")
                    chosen = int(state["best_epoch"])
                    if not 1 <= chosen <= len(history):
                        raise ValueError("invalid best_epoch")
                    selected = history[chosen - 1]
                    if int(selected["epoch"]) != chosen:
                        raise ValueError("best_epoch does not match history row")
                    if abs(float(selected["val_loss"]) - float(state["best_val_loss"])) > 1e-3:
                        raise ValueError("best validation loss inconsistent with history")
                    if abs(float(selected["val_auroc"]) - float(state["best_val_auroc"])) > 1e-4:
                        raise ValueError("AUROC at best epoch inconsistent with history")
                    runs.append({
                        "category": category, "mode": mode, "seed": seed,
                        "best_epoch": chosen,
                        "best_val_loss": state["best_val_loss"],
                        "best_val_auroc": state["best_val_auroc"],
                        "epochs_run": state["epoch"],
                        "train_count": state["train_count"],
                        "val_count": state["val_count"],
                        "protocol": state["protocol"],
                        "git_commit_at_training": state.get("git_commit", ""),
                    })
                except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
                    errors.append(f"{name}: {exc}")

    if errors:
        print("RESULT VERIFICATION FAILED:")
        for problem in errors:
            print(" -", problem)
        raise SystemExit(1)

    assert len(runs) == 27
    if write_summary:
        path = RESULTS / "summary.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
        print(f"Wrote {path.relative_to(ROOT)}")

    print("Verified 27 complete runs and 3 split CSVs.")
    print("Mean +/- population SD (ddof=0) of validation AUROC across 3 seeds:")
    for mode in MODES:
        scores_by_category = []
        for cat in CATEGORIES:
            values = [float(r["best_val_auroc"]) for r in runs if r["category"] == cat and r["mode"] == mode]
            print(f"  {mode:9s} {cat:10s} {mean(values):.4f} +/- {pstdev(values):.4f}")
            scores_by_category.append(values)
        per_seed_average = [mean(x[i] for x in scores_by_category) for i in range(3)]
        print(f"  {mode:9s} {'3-cat mean':10s} {mean(per_seed_average):.4f} +/- {pstdev(per_seed_average):.4f}")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify without writing summary.csv")
    args = parser.parse_args()
    verify_results(write_summary=not args.check_only)
