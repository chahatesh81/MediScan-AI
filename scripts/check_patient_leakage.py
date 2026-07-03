import re

import pandas as pd

from backend.app.core.config import PROCESSED_DATA_DIR


MANIFEST_FILE = PROCESSED_DATA_DIR / "clean_split_manifest.csv"
REPORT_FILE = PROCESSED_DATA_DIR / "patient_leakage_report.csv"


def extract_patient_id(file_name: str, class_name: str) -> str:
    if class_name == "PNEUMONIA":
        match = re.match(r"(person\d+)", file_name)

        if match:
            return match.group(1)

    # NORMAL filenames do not provide a reliable patient ID.
    # Use the full filename so we do not invent patient groupings.
    return f"unknown::{file_name}"


def main() -> None:
    df = pd.read_csv(MANIFEST_FILE)

    df["patient_id"] = df.apply(
        lambda row: extract_patient_id(
            row["file_name"],
            row["class_name"],
        ),
        axis=1,
    )

    pneumonia_df = df[df["class_name"] == "PNEUMONIA"].copy()

    patient_split_counts = (
        pneumonia_df.groupby("patient_id")["final_split"]
        .nunique()
    )

    leaking_patients = patient_split_counts[
        patient_split_counts > 1
    ].index

    leakage_df = pneumonia_df[
        pneumonia_df["patient_id"].isin(leaking_patients)
    ].copy()

    leakage_df = leakage_df.sort_values(
        ["patient_id", "final_split", "path"]
    )

    leakage_df.to_csv(REPORT_FILE, index=False)

    print("=" * 60)
    print("MEDISCAN AI — PATIENT-LEVEL LEAKAGE AUDIT")
    print("=" * 60)
    print(f"Images checked:              {len(df):,}")
    print(f"Pneumonia images checked:    {len(pneumonia_df):,}")
    print(f"Unique pneumonia patients:   {pneumonia_df['patient_id'].nunique():,}")
    print(f"Cross-split patient IDs:     {len(leaking_patients):,}")
    print(f"Images involved in leakage:  {len(leakage_df):,}")
    print(f"Report saved: {REPORT_FILE}")

    if len(leaking_patients) > 0:
        print("\nLeakage by split combination:")

        combinations = (
            leakage_df.groupby("patient_id")["final_split"]
            .apply(lambda values: " + ".join(sorted(set(values))))
            .value_counts()
        )

        print(combinations)


if __name__ == "__main__":
    main()