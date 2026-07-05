from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.xray_preprocessing import (
    crop_foreground,
    preprocess_xray,
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_manifest.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "xray_preprocessing_audit.png"
)

SAMPLES_PER_CLASS = 6
RANDOM_SEED = 42


def load_grayscale(
    image_path: Path,
) -> np.ndarray:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def select_samples() -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # Use TRAIN only.
    # Do not use test images to tune preprocessing.
    train_manifest = manifest[
        manifest["final_split"] == "train"
    ].copy()

    selected_groups = []

    for class_name in [
        "NORMAL",
        "PNEUMONIA",
    ]:
        group = train_manifest[
            train_manifest["class_name"]
            == class_name
        ]

        if len(group) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Not enough {class_name} images."
            )

        sampled = group.sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_SEED,
        )

        selected_groups.append(
            sampled
        )

    return pd.concat(
        selected_groups,
        ignore_index=True,
    )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "X-RAY PREPROCESSING VISUAL AUDIT"
    )
    print("=" * 70)

    selected = select_samples()

    print(
        f"Selected images: {len(selected)}"
    )

    figure, axes = plt.subplots(
        nrows=len(selected),
        ncols=3,
        figsize=(12, 4 * len(selected)),
    )

    retained_ratios = []

    for index, row in selected.iterrows():
        image_path = (
            IMAGE_ROOT
            / row["path"]
        )

        original = load_grayscale(
            image_path
        )

        cropped, bbox = crop_foreground(
            original
        )

        processed, metadata = preprocess_xray(
            original,
            target_size=224,
            return_metadata=True,
        )

        retained_ratios.append(
            metadata["retained_area_ratio"]
        )

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = bbox

        preview = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2RGB,
        )

        cv2.rectangle(
            preview,
            (x_min, y_min),
            (x_max - 1, y_max - 1),
            (255, 255, 255),
            max(
                1,
                original.shape[1] // 300,
            ),
        )

        axes[index, 0].imshow(
            preview
        )

        axes[index, 0].set_title(
            f"{row['class_name']} — Original\n"
            f"{row['file_name']}"
        )

        axes[index, 1].imshow(
            cropped,
            cmap="gray",
        )

        axes[index, 1].set_title(
            "Detected foreground crop\n"
            f"Retained: "
            f"{metadata['retained_area_ratio']:.1%}"
        )

        axes[index, 2].imshow(
            processed
        )

        axes[index, 2].set_title(
            "Final model input\n"
            "224 × 224, aspect ratio preserved"
        )

        for column in range(3):
            axes[index, column].axis(
                "off"
            )

        print(
            f"[{index + 1:2d}/{len(selected)}] "
            f"{row['class_name']:9s} | "
            f"original="
            f"{original.shape[1]}x"
            f"{original.shape[0]} | "
            f"crop="
            f"{cropped.shape[1]}x"
            f"{cropped.shape[0]} | "
            f"retained="
            f"{metadata['retained_area_ratio']:.3f}"
        )

    figure.suptitle(
        "MediScan AI — Artifact-Aware "
        "Preprocessing Audit",
        fontsize=16,
        y=1.0,
    )

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    retained_ratios = np.asarray(
        retained_ratios,
        dtype=np.float32,
    )

    print("\nSummary:")
    print(
        f"Minimum retained area: "
        f"{retained_ratios.min():.3f}"
    )
    print(
        f"Mean retained area:    "
        f"{retained_ratios.mean():.3f}"
    )
    print(
        f"Maximum retained area: "
        f"{retained_ratios.max():.3f}"
    )

    print(
        f"\nGenerated:\n{OUTPUT_PATH}"
    )

    print(
        "\nX-RAY PREPROCESSING "
        "VISUAL AUDIT STATUS: READY"
    )


if __name__ == "__main__":
    main()