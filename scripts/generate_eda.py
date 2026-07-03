from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from backend.app.core.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
)


MANIFEST_FILE = PROCESSED_DATA_DIR / "final_manifest.csv"
DATASET_DIR = RAW_DATA_DIR / "chest_xray"


def save_class_distribution(df: pd.DataFrame) -> None:
    counts = pd.crosstab(
        df["final_split"],
        df["class_name"],
    )

    ax = counts.plot(
        kind="bar",
        figsize=(9, 6),
    )

    ax.set_title("MediScan AI — Class Distribution")
    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_dimension_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["width"],
        df["height"],
        alpha=0.2,
        s=10,
    )

    plt.title("MediScan AI — Image Dimension Distribution")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "image_dimensions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_sample_grid(df: pd.DataFrame) -> None:
    samples = []

    for class_name in ("NORMAL", "PNEUMONIA"):
        class_samples = (
            df[
                (df["final_split"] == "train")
                & (df["class_name"] == class_name)
            ]
            .sample(
                n=4,
                random_state=RANDOM_SEED,
            )
        )

        samples.append(class_samples)

    sample_df = pd.concat(samples)

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
    )

    for axis, (_, row) in zip(
        axes.flatten(),
        sample_df.iterrows(),
    ):
        image_path = DATASET_DIR / row["path"]

        with Image.open(image_path) as image:
            axis.imshow(image, cmap="gray")

        axis.set_title(row["class_name"])
        axis.axis("off")

    figure.suptitle(
        "MediScan AI — Chest X-Ray Samples",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "xray_samples.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(MANIFEST_FILE)

    save_class_distribution(df)
    save_dimension_distribution(df)
    save_sample_grid(df)

    print("Generated EDA figures:")

    for figure_path in sorted(
        FIGURES_DIR.glob("*.png")
    ):
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()