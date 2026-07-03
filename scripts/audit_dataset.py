from pathlib import Path

import pandas as pd
from PIL import Image

from backend.app.core.config import RAW_DATA_DIR


DATASET_DIR = RAW_DATA_DIR / "chest_xray"
OUTPUT_FILE = RAW_DATA_DIR.parent / "processed" / "dataset_inventory.csv"

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def build_inventory() -> pd.DataFrame:
    records = []

    for split in ("train", "val", "test"):
        split_dir = DATASET_DIR / split

        for class_name in ("NORMAL", "PNEUMONIA"):
            class_dir = split_dir / class_name

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                record = {
                    "path": str(image_path.relative_to(DATASET_DIR)),
                    "split": split,
                    "class_name": class_name,
                    "file_name": image_path.name,
                    "file_size_bytes": image_path.stat().st_size,
                    "width": None,
                    "height": None,
                    "mode": None,
                    "format": None,
                    "is_valid": False,
                    "error": None,
                }

                try:
                    with Image.open(image_path) as image:
                        image.verify()

                    with Image.open(image_path) as image:
                        record["width"] = image.width
                        record["height"] = image.height
                        record["mode"] = image.mode
                        record["format"] = image.format
                        record["is_valid"] = True

                except Exception as error:
                    record["error"] = str(error)

                records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    print("Building MediScan AI dataset inventory...")

    inventory = build_inventory()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_FILE, index=False)

    print(f"Total images:   {len(inventory):,}")
    print(f"Valid images:   {inventory['is_valid'].sum():,}")
    print(f"Invalid images: {(~inventory['is_valid']).sum():,}")
    print(f"Inventory:      {OUTPUT_FILE}")

    print("\nImages by split and class:")
    print(
        inventory.groupby(
            ["split", "class_name"]
        ).size().unstack(fill_value=0)
    )


if __name__ == "__main__":
    main()