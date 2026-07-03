import pandas as pd

from backend.app.core.config import PROCESSED_DATA_DIR


MANIFEST_FILE = PROCESSED_DATA_DIR / "final_manifest.csv"


def main() -> None:
    df = pd.read_csv(MANIFEST_FILE)

    pneumonia = df[
        df["class_name"] == "PNEUMONIA"
    ].copy()

    patient_leakage = (
        pneumonia.dropna(subset=["patient_id"])
        .groupby("patient_id")["final_split"]
        .nunique()
        .gt(1)
        .sum()
    )

    hash_leakage = (
        df.groupby("sha256")["final_split"]
        .nunique()
        .gt(1)
        .sum()
    )

    duplicate_hashes = df["sha256"].duplicated().sum()

    required_splits = {"train", "val", "test"}
    actual_splits = set(df["final_split"].unique())

    checks = {
        "All three splits exist":
            actual_splits == required_splits,

        "No pneumonia patient leakage":
            patient_leakage == 0,

        "No exact-hash cross-split leakage":
            hash_leakage == 0,

        "No exact duplicate images":
            duplicate_hashes == 0,

        "No missing paths":
            df["path"].isna().sum() == 0,

        "No missing labels":
            df["class_name"].isna().sum() == 0,
    }

    print("=" * 60)
    print("MEDISCAN AI — FINAL DATASET VERIFICATION")
    print("=" * 60)

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

    print(f"\nTotal images:                {len(df):,}")
    print(f"Patient leakage sets:        {patient_leakage:,}")
    print(f"Cross-split hash leakage:    {hash_leakage:,}")
    print(f"Duplicate image rows:        {duplicate_hashes:,}")

    print("\nFinal split distribution:")
    print(
        pd.crosstab(
            df["final_split"],
            df["class_name"],
            margins=True,
        )
    )

    if not all(checks.values()):
        raise RuntimeError(
            "Final dataset verification failed."
        )

    print("\nFINAL DATASET STATUS: CLEAN AND LEAKAGE-SAFE")


if __name__ == "__main__":
    main()