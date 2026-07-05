import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.gradcam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)
from backend.app.ml.runtime import (
    configure_training_runtime,
)
from backend.app.services.data_pipeline import (
    build_dataset,
)


MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "mediscan_final.keras"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_test_metrics.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "gradcam_cases"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "gradcam_case_summary.json"
)


CASE_TYPES = {
    "true_positive": (1, 1),
    "true_negative": (0, 0),
    "false_positive": (0, 1),
    "false_negative": (1, 0),
}


def select_representative_cases(
    records: list[dict],
) -> dict[str, dict]:

    selected = {}

    for case_name, (
        required_true,
        required_pred,
    ) in CASE_TYPES.items():

        candidates = [
            record
            for record in records
            if record["true_label"] == required_true
            and record["predicted_label"] == required_pred
        ]

        if not candidates:
            raise RuntimeError(
                f"No samples found for {case_name}."
            )

        if case_name == "true_positive":
            selected[case_name] = max(
                candidates,
                key=lambda item: item["probability"],
            )

        elif case_name == "true_negative":
            selected[case_name] = min(
                candidates,
                key=lambda item: item["probability"],
            )

        elif case_name == "false_positive":
            selected[case_name] = max(
                candidates,
                key=lambda item: item["probability"],
            )

        else:
            selected[case_name] = min(
                candidates,
                key=lambda item: item["probability"],
            )

    return selected


def prepare_display_image(
    image: np.ndarray,
) -> np.ndarray:

    image = np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)

    return image


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:

    height, width = image.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap_uint8 = np.uint8(
        255 * resized_heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = (
        0.60 * image.astype(np.float32)
        + 0.40 * colored_heatmap.astype(np.float32)
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


def save_case_figure(
    case_name: str,
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    record: dict,
    explanation_mode: str,
    threshold: float,
) -> None:

    figure = plt.figure(
        figsize=(15, 5)
    )

    axis_1 = figure.add_subplot(
        1,
        3,
        1,
    )

    axis_1.imshow(image)
    axis_1.set_title("Original X-Ray")
    axis_1.axis("off")

    axis_2 = figure.add_subplot(
        1,
        3,
        2,
    )

    axis_2.imshow(
        heatmap,
        cmap="jet",
    )

    axis_2.set_title(
        f"Attribution Map\n"
        f"{explanation_mode}"
    )

    axis_2.axis("off")

    axis_3 = figure.add_subplot(
        1,
        3,
        3,
    )

    axis_3.imshow(overlay)

    axis_3.set_title(
        f"Overlay\n"
        f"P(Pneumonia) = "
        f"{record['probability']:.4f}"
    )

    axis_3.axis("off")

    true_name = (
        "PNEUMONIA"
        if record["true_label"] == 1
        else "NORMAL"
    )

    predicted_name = (
        "PNEUMONIA"
        if record["predicted_label"] == 1
        else "NORMAL"
    )

    figure.suptitle(
        f"{case_name.replace('_', ' ').title()} | "
        f"True: {true_name} | "
        f"Predicted: {predicted_name} | "
        f"Threshold: {threshold:.4f}",
        fontsize=13,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR / f"{case_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — GRAD-CAM CASE STUDY GENERATOR")
    print("=" * 70)

    configure_training_runtime()

    metrics = json.loads(
        METRICS_FILE.read_text()
    )

    threshold = float(
        metrics["threshold"]
    )

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )

    last_conv_layer = (
        find_last_conv_layer(model)
    )

    dataset = build_dataset(
        split="test",
        batch_size=1,
        shuffle=False,
    )

    records = []

    print("Scanning held-out test predictions...")

    for index, (
        images,
        labels,
    ) in enumerate(dataset):

        probability = float(
            model(
                images,
                training=False,
            )[0, 0].numpy()
        )

        true_label = int(
            labels[0].numpy()
        )

        predicted_label = int(
            probability >= threshold
        )

        records.append(
            {
                "index": index,
                "true_label": true_label,
                "predicted_label":
                    predicted_label,
                "probability": probability,
                "image": images[0].numpy(),
            }
        )

    selected_cases = (
        select_representative_cases(
            records
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model": str(MODEL_FILE),
        "threshold": threshold,
        "last_conv_layer":
            last_conv_layer,
        "cases": {},
    }

    print("\nGenerating explanations...")

    for case_name, record in (
        selected_cases.items()
    ):

        image_batch = tf.convert_to_tensor(
            record["image"][
                np.newaxis,
                ...
            ],
            dtype=tf.float32,
        )

        heatmap, explanation_mode = (
            generate_gradcam_heatmap(
                model=model,
                image_batch=image_batch,
                last_conv_layer_name=
                    last_conv_layer,
                return_mode=True,
            )
        )

        image = prepare_display_image(
            record["image"]
        )

        overlay = create_overlay(
            image,
            heatmap,
        )

        save_case_figure(
            case_name=case_name,
            image=image,
            heatmap=heatmap,
            overlay=overlay,
            record=record,
            explanation_mode=
                explanation_mode,
            threshold=threshold,
        )

        summary["cases"][case_name] = {
            "dataset_index":
                record["index"],
            "true_label":
                record["true_label"],
            "predicted_label":
                record["predicted_label"],
            "probability":
                record["probability"],
            "explanation_mode":
                explanation_mode,
            "heatmap_min":
                float(heatmap.min()),
            "heatmap_max":
                float(heatmap.max()),
        }

        print(
            f"{case_name:16s} | "
            f"index={record['index']:3d} | "
            f"probability="
            f"{record['probability']:.6f} | "
            f"{explanation_mode}"
        )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\nGenerated:")

    for case_name in CASE_TYPES:
        print(
            f"  {OUTPUT_DIR / (case_name + '.png')}"
        )

    print(f"\nSummary: {SUMMARY_FILE}")

    print(
        "\nGRAD-CAM CASE STUDY STATUS: READY"
    )


if __name__ == "__main__":
    main()