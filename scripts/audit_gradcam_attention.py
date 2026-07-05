from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.gradcam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediscan_final.keras"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_test_predictions.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_manifest.csv"
)

# Correct dataset location
IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "gradcam_attention_audit.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "gradcam_attention_audit_summary.txt"
)

IMAGE_SIZE = (224, 224)
MAX_SAMPLES_PER_GROUP = 25
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


def classify_case(
    true_label: int,
    predicted_label: int,
) -> str:
    if true_label == 1 and predicted_label == 1:
        return "true_positive"

    if true_label == 0 and predicted_label == 0:
        return "true_negative"

    if true_label == 0 and predicted_label == 1:
        return "false_positive"

    return "false_negative"


def calculate_attention_regions(
    heatmap: np.ndarray,
) -> dict[str, float]:
    """
    Calculate geometric attention diagnostics.

    These regions are engineering proxies only.
    They are not anatomical lung masks.
    """

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

    peak_in_border = float(
        border_mask[
            peak_y,
            peak_x,
        ]
    )

    return {
        "border_energy_ratio": (
            border_energy / total_energy
        ),
        "thorax_energy_ratio": (
            thorax_energy / total_energy
        ),
        "peak_in_border": peak_in_border,
    }


def build_test_table() -> pd.DataFrame:
    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_manifest = (
        manifest[
            manifest["final_split"] == "test"
        ]
        .reset_index(drop=True)
        .copy()
    )

    print(
        f"Prediction rows: {len(predictions):,}"
    )

    print(
        f"Test manifest rows: "
        f"{len(test_manifest):,}"
    )

    if len(predictions) != len(test_manifest):
        raise RuntimeError(
            "Prediction count does not match "
            "test manifest count."
        )

    expected_labels = (
        test_manifest["class_name"]
        .map(
            {
                "NORMAL": 0,
                "PNEUMONIA": 1,
            }
        )
        .to_numpy(dtype=np.int32)
    )

    prediction_labels = (
        predictions["true_label"]
        .to_numpy(dtype=np.int32)
    )

    labels_match = np.array_equal(
        expected_labels,
        prediction_labels,
    )

    print(
        f"Row-order label check: "
        f"{'PASS' if labels_match else 'FAIL'}"
    )

    if not labels_match:
        mismatch_indices = np.flatnonzero(
            expected_labels
            != prediction_labels
        )

        print(
            "First mismatching indices:",
            mismatch_indices[:10].tolist(),
        )

        raise RuntimeError(
            "Prediction rows are not aligned "
            "with test manifest rows. "
            "Audit stopped."
        )

    combined = test_manifest.copy()

    combined["true_label"] = (
        predictions["true_label"]
        .to_numpy()
    )

    combined["probability"] = (
        predictions["probability"]
        .to_numpy()
    )

    combined["predicted_label"] = (
        predictions["predicted_label"]
        .to_numpy()
    )

    combined["case_type"] = combined.apply(
        lambda row: classify_case(
            int(row["true_label"]),
            int(row["predicted_label"]),
        ),
        axis=1,
    )

    combined["image_path"] = (
        combined["path"]
        .apply(
            lambda relative_path: str(
                IMAGE_ROOT
                / relative_path
            )
        )
    )

    missing_files = [
        path
        for path in combined["image_path"]
        if not Path(path).is_file()
    ]

    print(
        f"Missing image files: "
        f"{len(missing_files)}"
    )

    if missing_files:
        print(
            "First missing path:"
        )

        print(
            missing_files[0]
        )

        raise FileNotFoundError(
            "One or more test images "
            "could not be found."
        )

    return combined


def sample_cases(
    test_table: pd.DataFrame,
) -> pd.DataFrame:
    selected_groups = []

    print("\nAvailable cases:")

    case_order = [
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ]

    for case_type in case_order:
        group = test_table[
            test_table["case_type"]
            == case_type
        ]

        available = len(group)

        sample_count = min(
            MAX_SAMPLES_PER_GROUP,
            available,
        )

        print(
            f"{case_type:16s}: "
            f"{available:3d} available, "
            f"{sample_count:2d} selected"
        )

        if sample_count == 0:
            continue

        sampled = group.sample(
            n=sample_count,
            random_state=RANDOM_SEED,
        )

        selected_groups.append(
            sampled
        )

    if not selected_groups:
        raise RuntimeError(
            "No cases were selected."
        )

    return pd.concat(
        selected_groups,
        ignore_index=True,
    )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "GRAD-CAM ATTENTION AUDIT"
    )
    print("=" * 70)

    np.random.seed(
        RANDOM_SEED
    )

    print(
        "\nBuilding aligned test table..."
    )

    test_table = build_test_table()

    selected = sample_cases(
        test_table
    )

    print(
        f"\nTotal images selected: "
        f"{len(selected)}"
    )

    print(
        f"\nLoading model:"
        f"\n{MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    last_conv_layer = (
        find_last_conv_layer(
            model
        )
    )

    print(
        f"Last conv layer: "
        f"{last_conv_layer}"
    )

    records = []

    print("\nAuditing attention...")

    total_selected = len(selected)

    for position, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        image_path = Path(
            row["image_path"]
        )

        image_batch = load_image(
            image_path
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

        metrics = (
            calculate_attention_regions(
                heatmap
            )
        )

        records.append(
            {
                "image_path": str(
                    image_path
                ),
                "relative_path": row["path"],
                "patient_id": row["patient_id"],
                "case_type": row["case_type"],
                "true_label": int(
                    row["true_label"]
                ),
                "predicted_label": int(
                    row["predicted_label"]
                ),
                "probability": float(
                    row["probability"]
                ),
                "explanation_mode": (
                    explanation_mode
                ),
                **metrics,
            }
        )

        print(
            f"[{position:3d}/"
            f"{total_selected}] "
            f"{row['case_type']:16s} | "
            f"border="
            f"{metrics['border_energy_ratio']:.3f} | "
            f"thorax="
            f"{metrics['thorax_energy_ratio']:.3f} | "
            f"peak_border="
            f"{int(metrics['peak_in_border'])}"
        )

    results = pd.DataFrame(
        records
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = (
        results
        .groupby("case_type")
        .agg(
            samples=(
                "image_path",
                "count",
            ),
            mean_border_energy=(
                "border_energy_ratio",
                "mean",
            ),
            mean_thorax_energy=(
                "thorax_energy_ratio",
                "mean",
            ),
            border_peak_rate=(
                "peak_in_border",
                "mean",
            ),
        )
        .round(4)
    )

    mode_counts = (
        results[
            "explanation_mode"
        ]
        .value_counts()
    )

    overall_border = float(
        results[
            "border_energy_ratio"
        ].mean()
    )

    overall_thorax = float(
        results[
            "thorax_energy_ratio"
        ].mean()
    )

    overall_peak_rate = float(
        results[
            "peak_in_border"
        ].mean()
    )

    report = [
        "=" * 70,
        (
            "MEDISCAN AI — "
            "GRAD-CAM ATTENTION AUDIT SUMMARY"
        ),
        "=" * 70,
        "",
        summary.to_string(),
        "",
        "Explanation modes:",
        mode_counts.to_string(),
        "",
        "Overall:",
        (
            f"Mean border energy: "
            f"{overall_border:.4f}"
        ),
        (
            f"Mean thorax energy: "
            f"{overall_thorax:.4f}"
        ),
        (
            f"Border peak rate: "
            f"{overall_peak_rate:.4f}"
        ),
        "",
        "Interpretation:",
        (
            "Higher border energy and border "
            "peak rate indicate greater "
            "shortcut-learning risk."
        ),
        (
            "The thoracic region is a geometric "
            "proxy, not an anatomical lung mask."
        ),
    ]

    report_text = "\n".join(
        report
    )

    SUMMARY_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print("\n")
    print(report_text)

    print(
        f"\nCSV:     {OUTPUT_PATH}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        "\nGRAD-CAM ATTENTION "
        "AUDIT STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()