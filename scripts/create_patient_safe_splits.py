import re

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.core.config import PROCESSED_DATA_DIR, RANDOM_SEED


INPUT_FILE = PROCESSED_DATA_DIR / "clean_split_manifest.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "final_manifest.csv"
EXCLUDED_FILE = PROCESSED_DATA_DIR / "patient_overlap_excluded.csv"

VALIDATION_SIZE = 0.15


def extract_patient_id(file_name: str) -> str | None:
    match = re.match(r"(person\d+)", file_name)
    return match.group(1) if match else None


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    df["patient_id"] = df.apply(
        lambda row: (
            extract_patient_id(row["file_name"])
            if row["class_name"] == "PNEUMONIA"
            else None
        ),
        axis=1,
    )

    official_test = df[df["final_split"] == "test"].copy()
    development = df[df["final_split"] != "test"].copy()

    test_pneumonia_patients = set(
        official_test.loc[
            official_test["class_name"] == "PNEUMONIA",
            "patient_id",
        ].dropna()
    )

    overlap_mask = (
        (development["class_name"] == "PNEUMONIA")
        & development["patient_id"].isin(test_pneumonia_patients)
    )

    excluded = development[overlap_mask].copy()
    development = development[~overlap_mask].copy()

    pneumonia = development[
        development["class_name"] == "PNEUMONIA"
    ].copy()

    normal = development[
        development["class_name"] == "NORMAL"
    ].copy()

    pneumonia_patients = pneumonia[
        "patient_id"
    ].dropna().unique()

    train_patients, val_patients = train_test_split(
        pneumonia_patients,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )

    pneumonia_train = pneumonia[
        pneumonia["patient_id"].isin(train_patients)
    ].copy()

    pneumonia_val = pneumonia[
        pneumonia["patient_id"].isin(val_patients)
    ].copy()

    normal_train, normal_val = train_test_split(
        normal,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )

    train_df = pd.concat(
        [pneumonia_train, normal_train],
        ignore_index=True,
    )

    val_df = pd.concat(
        [pneumonia_val, normal_val],
        ignore_index=True,
    )

    train_df["final_split"] = "train"
    val_df["final_split"] = "val"
    official_test["final_split"] = "test"

    final_df = pd.concat(
        [train_df, val_df, official_test],
        ignore_index=True,
    )

    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)
    excluded.to_csv(EXCLUDED_FILE, index=False)

    pneumonia_check = final_df[
        final_df["class_name"] == "PNEUMONIA"
    ]

    remaining_patient_leakage = (
        pneumonia_check.groupby("patient_id")["final_split"]
        .nunique()
        .gt(1)
        .sum()
    )

    remaining_hash_leakage = (
        final_df.groupby("sha256")["final_split"]
        .nunique()
        .gt(1)
        .sum()
    )

    print("=" * 60)
    print("MEDISCAN AI — PATIENT-SAFE FINAL SPLITS")
    print("=" * 60)

    print(f"Images excluded due to test-patient overlap: {len(excluded):,}")
    print(f"Final usable images:                        {len(final_df):,}")
    print(f"Remaining pneumonia patient leakage:        {remaining_patient_leakage:,}")
    print(f"Remaining exact-hash leakage:               {remaining_hash_leakage:,}")

    print("\nFinal split counts:")
    print(
        pd.crosstab(
            final_df["final_split"],
            final_df["class_name"],
            margins=True,
        )
    )

    print(f"\nFinal manifest: {OUTPUT_FILE}")
    print(f"Exclusion audit: {EXCLUDED_FILE}")


if __name__ == "__main__":
    main()