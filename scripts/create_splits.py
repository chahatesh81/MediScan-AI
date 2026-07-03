import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.core.config import PROCESSED_DATA_DIR, RANDOM_SEED


INVENTORY_FILE = PROCESSED_DATA_DIR / "dataset_inventory.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "split_manifest.csv"

VALIDATION_SIZE = 0.15


def main() -> None:
    df = pd.read_csv(INVENTORY_FILE)
    df = df[df["is_valid"] == True].copy()

    original_train = df[df["split"] == "train"].copy()
    original_val = df[df["split"] == "val"].copy()
    official_test = df[df["split"] == "test"].copy()

    train_df, val_df = train_test_split(
        original_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=original_train["class_name"],
    )

    train_df["final_split"] = "train"
    val_df["final_split"] = "val"
    official_test["final_split"] = "test"

    final_manifest = pd.concat(
        [train_df, val_df, official_test],
        ignore_index=True,
    )

    final_manifest = final_manifest.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_manifest.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("MEDISCAN AI — FINAL DATA SPLITS")
    print("=" * 60)

    print(
        pd.crosstab(
            final_manifest["final_split"],
            final_manifest["class_name"],
            margins=True,
        )
    )

    print(f"\nOriginal validation images excluded: {len(original_val)}")
    print(f"Total usable images: {len(final_manifest):,}")
    print(f"Manifest saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()