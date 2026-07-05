from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.gradcam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)


V1_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_cnn_best.keras"
)

V2_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_cnn_v2_best.keras"
)

CACHE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_v2_cache_manifest.csv"
)

V1_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "attention_v1_v2_paired_comparison.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "attention_v1_v2_paired_summary.txt"
)

IMAGE_SIZE = (224, 224)
SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42


def load_image(
    image_path: Path,
) -> tf.Tensor:
    image_bytes = tf.io.read_file(
        str(image_path)
    )

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return tf.expand_dims(
        image,
        axis=0,
    )


def calculate_attention_regions(
    heatmap: np.ndarray,
) -> dict[str, float]:
    height, width = heatmap.shape

    border_y = max(
        1,
        int(height * 0.20),
    )

    border_x = max(
        1,
        int(width * 0.20),
    )

    border_mask = np.ones(
        (height, width),
        dtype=bool,
    )

    border_mask[
        border_y : height - border_y,
        border_x : width - border_x,
    ] = False

    thorax_mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    thorax_mask[
        int(height * 0.15) : int(height * 0.90),
        int(width * 0.15) : int(width * 0.85),
    ] = True

    total_energy = float(
        np.sum(heatmap)
    )

    if total_energy <= 0.0:
        return {
            "border_energy_ratio": 0.0,
            "thorax_energy_ratio": 0.0,
            "peak_in_border": 0.0,
        }

    border_energy = float(
        np.sum(
            heatmap[border_mask]
        )
    )

    thorax_energy = float(
        np.sum(
            heatmap[thorax_mask]
        )
    )

    peak_y, peak_x = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape,
    )

    return {
        "border_energy_ratio": (
            border_energy / total_energy
        ),
        "thorax_energy_ratio": (
            thorax_energy / total_energy
        ),
        "peak_in_border": float(
            border_mask[
                peak_y,
                peak_x,
            ]
        ),
    }


def build_paired_sample() -> pd.DataFrame:
    manifest = pd.read_csv(
        CACHE_MANIFEST_PATH
    )

    validation = (
        manifest[
            manifest["final_split"] == "val"
        ]
        .reset_index(drop=True)
        .copy()
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    if len(validation) != 713:
        raise RuntimeError(
            "Expected 713 validation rows, "
            f"found {len(validation)}."
        )

    selected_groups = []

    for label, class_name in [
        (0, "NORMAL"),
        (1, "PNEUMONIA"),
    ]:
        group = validation[
            validation["label"] == label
        ]

        sample_count = min(
            SAMPLES_PER_CLASS,
            len(group),
        )

        sampled = group.sample(
            n=sample_count,
            random_state=RANDOM_SEED,
        )

        selected_groups.append(
            sampled
        )

        print(
            f"{class_name:10s}: "
            f"{sample_count} selected"
        )

    selected = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    selected["v1_image_path"] = (
        selected["source_path"]
        .apply(
            lambda path: str(
                V1_IMAGE_ROOT / path
            )
        )
    )

    selected["v2_image_path"] = (
        selected["cache_path"]
        .apply(
            lambda path: str(
                PROJECT_ROOT / path
            )
        )
    )

    for column in [
        "v1_image_path",
        "v2_image_path",
    ]:
        missing = [
            path
            for path in selected[column]
            if not Path(path).is_file()
        ]

        print(
            f"Missing {column}: "
            f"{len(missing)}"
        )

        if missing:
            print(
                f"First missing: "
                f"{missing[0]}"
            )

            raise FileNotFoundError(
                f"Missing files in {column}."
            )

    return selected


def audit_model_image(
    model: tf.keras.Model,
    image_path: Path,
    last_conv_layer: str,
) -> tuple[
    float,
    str,
    dict[str, float],
]:
    image_batch = load_image(
        image_path
    )

    probability = float(
        model(
            image_batch,
            training=False,
        )
        .numpy()
        .reshape(-1)[0]
    )

    (
        heatmap,
        explanation_mode,
    ) = generate_gradcam_heatmap(
        model=model,
        image_batch=image_batch,
        last_conv_layer_name=(
            last_conv_layer
        ),
        return_mode=True,
    )

    metrics = calculate_attention_regions(
        heatmap
    )

    return (
        probability,
        explanation_mode,
        metrics,
    )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "PAIRED V1 VS V2 ATTENTION COMPARISON"
    )
    print("=" * 70)

    np.random.seed(
        RANDOM_SEED
    )

    selected = build_paired_sample()

    print(
        f"\nTotal paired images: "
        f"{len(selected)}"
    )

    print(
        "\nLoading v1 model..."
    )

    v1_model = tf.keras.models.load_model(
        V1_MODEL_PATH,
        compile=False,
    )

    print(
        "Loading v2 model..."
    )

    v2_model = tf.keras.models.load_model(
        V2_MODEL_PATH,
        compile=False,
    )

    v1_last_conv = find_last_conv_layer(
        v1_model
    )

    v2_last_conv = find_last_conv_layer(
        v2_model
    )

    print(
        f"V1 last conv layer: "
        f"{v1_last_conv}"
    )

    print(
        f"V2 last conv layer: "
        f"{v2_last_conv}"
    )

    records = []

    total = len(selected)

    print(
        "\nRunning paired audit..."
    )

    for position, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        (
            v1_probability,
            v1_mode,
            v1_metrics,
        ) = audit_model_image(
            model=v1_model,
            image_path=Path(
                row["v1_image_path"]
            ),
            last_conv_layer=(
                v1_last_conv
            ),
        )

        (
            v2_probability,
            v2_mode,
            v2_metrics,
        ) = audit_model_image(
            model=v2_model,
            image_path=Path(
                row["v2_image_path"]
            ),
            last_conv_layer=(
                v2_last_conv
            ),
        )

        records.append(
            {
                "source_sha256": (
                    row["source_sha256"]
                ),
                "source_path": (
                    row["source_path"]
                ),
                "cache_path": (
                    row["cache_path"]
                ),
                "class_name": (
                    row["class_name"]
                ),
                "true_label": int(
                    row["label"]
                ),
                "retained_area_ratio": float(
                    row[
                        "retained_area_ratio"
                    ]
                ),
                "v1_probability": (
                    v1_probability
                ),
                "v2_probability": (
                    v2_probability
                ),
                "v1_explanation_mode": (
                    v1_mode
                ),
                "v2_explanation_mode": (
                    v2_mode
                ),
                "v1_border_energy": (
                    v1_metrics[
                        "border_energy_ratio"
                    ]
                ),
                "v2_border_energy": (
                    v2_metrics[
                        "border_energy_ratio"
                    ]
                ),
                "v1_thorax_energy": (
                    v1_metrics[
                        "thorax_energy_ratio"
                    ]
                ),
                "v2_thorax_energy": (
                    v2_metrics[
                        "thorax_energy_ratio"
                    ]
                ),
                "v1_peak_in_border": (
                    v1_metrics[
                        "peak_in_border"
                    ]
                ),
                "v2_peak_in_border": (
                    v2_metrics[
                        "peak_in_border"
                    ]
                ),
            }
        )

        if (
            position % 10 == 0
            or position == total
        ):
            print(
                f"{position:3d}/{total} complete"
            )

    results = pd.DataFrame(
        records
    )

    results[
        "border_energy_change"
    ] = (
        results["v2_border_energy"]
        - results["v1_border_energy"]
    )

    results[
        "thorax_energy_change"
    ] = (
        results["v2_thorax_energy"]
        - results["v1_thorax_energy"]
    )

    results[
        "border_improved"
    ] = (
        results["v2_border_energy"]
        < results["v1_border_energy"]
    )

    results[
        "thorax_improved"
    ] = (
        results["v2_thorax_energy"]
        > results["v1_thorax_energy"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = {
        "samples": len(results),
        "v1_mean_border_energy": float(
            results[
                "v1_border_energy"
            ].mean()
        ),
        "v2_mean_border_energy": float(
            results[
                "v2_border_energy"
            ].mean()
        ),
        "v1_mean_thorax_energy": float(
            results[
                "v1_thorax_energy"
            ].mean()
        ),
        "v2_mean_thorax_energy": float(
            results[
                "v2_thorax_energy"
            ].mean()
        ),
        "v1_border_peak_rate": float(
            results[
                "v1_peak_in_border"
            ].mean()
        ),
        "v2_border_peak_rate": float(
            results[
                "v2_peak_in_border"
            ].mean()
        ),
        "border_energy_improved_rate": float(
            results[
                "border_improved"
            ].mean()
        ),
        "thorax_energy_improved_rate": float(
            results[
                "thorax_improved"
            ].mean()
        ),
    }

    summary_lines = [
        "=" * 70,
        (
            "MEDISCAN AI — "
            "PAIRED V1 VS V2 ATTENTION SUMMARY"
        ),
        "=" * 70,
        "",
        (
            f"Paired samples:             "
            f"{summary['samples']}"
        ),
        "",
        (
            f"V1 mean border energy:      "
            f"{summary['v1_mean_border_energy']:.4f}"
        ),
        (
            f"V2 mean border energy:      "
            f"{summary['v2_mean_border_energy']:.4f}"
        ),
        "",
        (
            f"V1 mean thorax energy:      "
            f"{summary['v1_mean_thorax_energy']:.4f}"
        ),
        (
            f"V2 mean thorax energy:      "
            f"{summary['v2_mean_thorax_energy']:.4f}"
        ),
        "",
        (
            f"V1 border peak rate:        "
            f"{summary['v1_border_peak_rate']:.4f}"
        ),
        (
            f"V2 border peak rate:        "
            f"{summary['v2_border_peak_rate']:.4f}"
        ),
        "",
        (
            f"Border energy improved:     "
            f"{summary['border_energy_improved_rate']:.4f}"
        ),
        (
            f"Thorax energy improved:     "
            f"{summary['thorax_energy_improved_rate']:.4f}"
        ),
    ]

    summary_text = "\n".join(
        summary_lines
    )

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(
        "\n" + summary_text
    )

    print(
        f"\nCSV:     {OUTPUT_PATH}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        "\nPAIRED ATTENTION COMPARISON "
        "STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()