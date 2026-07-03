import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

from backend.app.core.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


DATASET_DIR = RAW_DATA_DIR / "chest_xray"
MANIFEST_FILE = PROCESSED_DATA_DIR / "split_manifest.csv"
REPORT_FILE = PROCESSED_DATA_DIR / "duplicate_report.csv"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def main() -> None:
    df = pd.read_csv(MANIFEST_FILE)

    print("Hashing 5,840 images...")

    df["sha256"] = [
        sha256_file(DATASET_DIR / relative_path)
        for relative_path in df["path"]
    ]

    hash_groups = defaultdict(list)

    for index, row in df.iterrows():
        hash_groups[row["sha256"]].append(index)

    duplicate_records = []

    for file_hash, indices in hash_groups.items():
        if len(indices) <= 1:
            continue

        group = df.loc[indices]
        splits = sorted(group["final_split"].unique())
        classes = sorted(group["class_name"].unique())

        for _, row in group.iterrows():
            duplicate_records.append(
                {
                    "sha256": file_hash,
                    "path": row["path"],
                    "final_split": row["final_split"],
                    "class_name": row["class_name"],
                    "duplicate_count": len(group),
                    "cross_split": len(splits) > 1,
                    "cross_class": len(classes) > 1,
                }
            )

    duplicate_df = pd.DataFrame(duplicate_records)

    if duplicate_df.empty:
        duplicate_df = pd.DataFrame(
            columns=[
                "sha256",
                "path",
                "final_split",
                "class_name",
                "duplicate_count",
                "cross_split",
                "cross_class",
            ]
        )

    duplicate_df.to_csv(REPORT_FILE, index=False)

    duplicate_hashes = duplicate_df["sha256"].nunique()
    cross_split_hashes = (
        duplicate_df.loc[
            duplicate_df["cross_split"] == True,
            "sha256",
        ].nunique()
    )
    cross_class_hashes = (
        duplicate_df.loc[
            duplicate_df["cross_class"] == True,
            "sha256",
        ].nunique()
    )

    print("\n" + "=" * 60)
    print("MEDISCAN AI — EXACT DUPLICATE AUDIT")
    print("=" * 60)
    print(f"Images checked:             {len(df):,}")
    print(f"Duplicate hash groups:      {duplicate_hashes:,}")
    print(f"Cross-split duplicate sets: {cross_split_hashes:,}")
    print(f"Cross-class duplicate sets: {cross_class_hashes:,}")
    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()