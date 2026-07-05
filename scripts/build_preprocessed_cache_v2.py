from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import pandas as pd

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.xray_preprocessing import preprocess_xray


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_manifest.csv"
)

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_v2_cache"
)

CACHE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_v2_cache_manifest.csv"
)

CACHE_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_v2_cache_metadata.json"
)

IMAGE_SIZE = 224
JPEG_QUALITY = 95


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "BASELINE V2 PREPROCESSED CACHE BUILDER"
    )
    print("=" * 70)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest = (
        manifest[
            manifest["final_split"].isin(
                [
                    "train",
                    "val",
                    "test",
                ]
            )
        ]
        .reset_index(drop=True)
        .copy()
    )

    expected_counts = {
        "train": 4024,
        "val": 713,
        "test": 618,
    }

    actual_counts = (
        manifest["final_split"]
        .value_counts()
        .to_dict()
    )

    if actual_counts != expected_counts:
        raise RuntimeError(
            "Unexpected split counts.\n"
            f"Expected: {expected_counts}\n"
            f"Actual:   {actual_counts}"
        )

    print(
        f"Images to preprocess: "
        f"{len(manifest):,}"
    )

    print(
        f"Cache root: "
        f"{CACHE_ROOT}"
    )

    if CACHE_ROOT.exists():
        print(
            "\nRemoving existing incomplete "
            "v2 cache..."
        )

        shutil.rmtree(
            CACHE_ROOT
        )

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    start_time = time.perf_counter()

    for index, row in manifest.iterrows():
        source_path = (
            SOURCE_ROOT
            / row["path"]
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source image missing: "
                f"{source_path}"
            )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Could not decode: "
                f"{source_path}"
            )

        (
            processed,
            metadata,
        ) = preprocess_xray(
            image,
            target_size=IMAGE_SIZE,
            return_metadata=True,
        )

        split_name = str(
            row["final_split"]
        )

        class_name = str(
            row["class_name"]
        )

        output_directory = (
            CACHE_ROOT
            / split_name
            / class_name
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_name = (
            f"{index:05d}_"
            f"{row['sha256'][:12]}.jpg"
        )

        output_path = (
            output_directory
            / output_name
        )

        success = cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                processed,
                cv2.COLOR_RGB2BGR,
            ),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY,
            ],
        )

        if not success:
            raise RuntimeError(
                f"Could not save: "
                f"{output_path}"
            )

        records.append(
            {
                "source_path": (
                    row["path"]
                ),
                "cache_path": str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "final_split": (
                    split_name
                ),
                "class_name": (
                    class_name
                ),
                "label": (
                    0
                    if class_name == "NORMAL"
                    else 1
                ),
                "patient_id": (
                    row["patient_id"]
                ),
                "source_sha256": (
                    row["sha256"]
                ),
                "original_width": (
                    metadata[
                        "original_width"
                    ]
                ),
                "original_height": (
                    metadata[
                        "original_height"
                    ]
                ),
                "cropped_width": (
                    metadata[
                        "cropped_width"
                    ]
                ),
                "cropped_height": (
                    metadata[
                        "cropped_height"
                    ]
                ),
                "retained_area_ratio": (
                    metadata[
                        "retained_area_ratio"
                    ]
                ),
            }
        )

        completed = index + 1

        if (
            completed % 250 == 0
            or completed == len(manifest)
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            rate = (
                completed / elapsed
            )

            print(
                f"[{completed:5d}/"
                f"{len(manifest)}] "
                f"{rate:.1f} images/s"
            )

    cache_manifest = pd.DataFrame(
        records
    )

    cache_manifest.to_csv(
        CACHE_MANIFEST_PATH,
        index=False,
    )

    missing_cache_files = [
        cache_path
        for cache_path
        in cache_manifest["cache_path"]
        if not (
            PROJECT_ROOT
            / cache_path
        ).is_file()
    ]

    if missing_cache_files:
        raise RuntimeError(
            "Cache verification failed. "
            f"Missing files: "
            f"{len(missing_cache_files)}"
        )

    cache_counts = (
        cache_manifest[
            "final_split"
        ]
        .value_counts()
        .to_dict()
    )

    if cache_counts != expected_counts:
        raise RuntimeError(
            "Cached split counts are incorrect."
        )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    metadata = {
        "pipeline_version": (
            "baseline_v2"
        ),
        "image_size": (
            IMAGE_SIZE
        ),
        "image_format": (
            "JPEG"
        ),
        "jpeg_quality": (
            JPEG_QUALITY
        ),
        "total_images": int(
            len(cache_manifest)
        ),
        "split_counts": {
            key: int(value)
            for key, value
            in cache_counts.items()
        },
        "mean_retained_area_ratio": float(
            cache_manifest[
                "retained_area_ratio"
            ].mean()
        ),
        "minimum_retained_area_ratio": float(
            cache_manifest[
                "retained_area_ratio"
            ].min()
        ),
        "maximum_retained_area_ratio": float(
            cache_manifest[
                "retained_area_ratio"
            ].max()
        ),
        "build_time_seconds": float(
            elapsed_time
        ),
    }

    CACHE_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("CACHE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Cached images:      "
        f"{len(cache_manifest):,}"
    )

    print(
        f"Train:              "
        f"{cache_counts['train']:,}"
    )

    print(
        f"Validation:         "
        f"{cache_counts['val']:,}"
    )

    print(
        f"Test:               "
        f"{cache_counts['test']:,}"
    )

    print(
        f"Mean retained area: "
        f"{metadata['mean_retained_area_ratio']:.4f}"
    )

    print(
        f"Minimum retained:   "
        f"{metadata['minimum_retained_area_ratio']:.4f}"
    )

    print(
        f"Build time:         "
        f"{elapsed_time / 60.0:.2f} minutes"
    )

    print(
        f"\nManifest:\n"
        f"{CACHE_MANIFEST_PATH}"
    )

    print(
        f"\nMetadata:\n"
        f"{CACHE_METADATA_PATH}"
    )

    print(
        "\nBASELINE V2 CACHE STATUS: READY"
    )


if __name__ == "__main__":
    main()