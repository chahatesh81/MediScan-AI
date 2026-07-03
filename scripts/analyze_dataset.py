from pathlib import Path

import pandas as pd

from backend.app.core.config import PROCESSED_DATA_DIR


INVENTORY_FILE = PROCESSED_DATA_DIR / "dataset_inventory.csv"
REPORT_FILE = PROCESSED_DATA_DIR / "dataset_summary.txt"


def main() -> None:
    df = pd.read_csv(INVENTORY_FILE)

    valid_df = df[df["is_valid"] == True].copy()

    split_counts = pd.crosstab(
        valid_df["split"],
        valid_df["class_name"],
        margins=True,
    )

    train_df = valid_df[valid_df["split"] == "train"]
    train_counts = train_df["class_name"].value_counts()

    normal_count = train_counts["NORMAL"]
    pneumonia_count = train_counts["PNEUMONIA"]
    imbalance_ratio = pneumonia_count / normal_count

    dimension_summary = valid_df.groupby("split").agg(
        image_count=("path", "count"),
        min_width=("width", "min"),
        max_width=("width", "max"),
        median_width=("width", "median"),
        min_height=("height", "min"),
        max_height=("height", "max"),
        median_height=("height", "median"),
    )

    mode_counts = valid_df["mode"].value_counts()

    report = f"""
MEDISCAN AI — DATASET SUMMARY
========================================

SPLIT AND CLASS COUNTS
{split_counts}

TRAINING CLASS IMBALANCE
NORMAL:     {normal_count}
PNEUMONIA:  {pneumonia_count}
Ratio:      {imbalance_ratio:.2f}:1

IMAGE DIMENSIONS
{dimension_summary}

IMAGE MODES
{mode_counts}

KEY FINDING
The original validation split contains only
{len(valid_df[valid_df["split"] == "val"])} images.

This split is too small for reliable model selection.
A new stratified validation split must be created
from the original training data.
""".strip()

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report)

    print(report)
    print(f"\nSaved report: {REPORT_FILE}")


if __name__ == "__main__":
    main()