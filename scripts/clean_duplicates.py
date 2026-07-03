import hashlib
from pathlib import Path

import pandas as pd

from backend.app.core.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


DATASET_DIR = RAW_DATA_DIR / "chest_xray"
MANIFEST_FILE = PROCESSED_DATA_DIR / "split_manifest.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "clean_split_manifest.csv"
REMOVED_FILE = PROCESSED_DATA_DIR / "removed_duplicates.csv"

SPLIT_PRIORITY = {
    "test": 0,
    "val": 1,
    "train": 2,
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def main() -> None:
    df = pd.read_csv(MANIFEST_FILE)

    print(f"Hashing {len(df):,} images...")

    df["sha256"] = [
        sha256_file(DATASET_DIR / path)
        for path in df["path"]
    ]

    df["split_priority"] = df["final_split"].map(SPLIT_PRIORITY)

    sorted_df = df.sort_values(
        by=["sha256", "split_priority", "path"]
    )

    keep_mask = ~sorted_df.duplicated(
        subset="sha256",
        keep="first",
    )

    clean_df = sorted_df[keep_mask].copy()
    removed_df = sorted_df[~keep_mask].copy()

    clean_df = clean_df.drop(
        columns=["split_priority"]
    ).reset_index(drop=True)

    removed_df = removed_df.drop(
        columns=["split_priority"]
    ).reset_index(drop=True)

    clean_df.to_csv(OUTPUT_FILE, index=False)
    removed_df.to_csv(REMOVED_FILE, index=False)

    remaining_cross_split = (
        clean_df.groupby("sha256")["final_split"]
        .nunique()
        .gt(1)
        .sum()
    )

    print("\n" + "=" * 60)
    print("MEDISCAN AI — DUPLICATE CLEANING")
    print("=" * 60)
    print(f"Original manifest:          {len(df):,}")
    print(f"Duplicate rows removed:     {len(removed_df):,}")
    print(f"Clean manifest:             {len(clean_df):,}")
    print(f"Remaining cross-split sets: {remaining_cross_split:,}")

    print("\nFinal split counts:")
    print(
        pd.crosstab(
            clean_df["final_split"],
            clean_df["class_name"],
            margins=True,
        )
    )

    print(f"\nClean manifest: {OUTPUT_FILE}")
    print(f"Removal audit: {REMOVED_FILE}")


if __name__ == "__main__":
    main()