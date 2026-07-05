import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.data_pipeline_v2_cached import (
    build_cached_dataset_v2,
)


SOURCE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "baseline_cnn_v2_best.keras"
)

FINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mediscan_final_v2.keras"
)

THRESHOLD_ANALYSIS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_v2_threshold_analysis.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

METRICS_FILE = (
    RESULTS_DIR
    / "final_test_v2_metrics.json"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "final_test_v2_predictions.csv"
)

EXPECTED_TEST_SAMPLES = 618


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "FINAL V2 HELD-OUT TEST EVALUATION"
    )
    print("=" * 70)

    configure_training_runtime()

    # -------------------------------------------------
    # Validate required artifacts
    # -------------------------------------------------

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(
            "V2 checkpoint not found: "
            f"{SOURCE_MODEL}"
        )

    if not THRESHOLD_ANALYSIS_FILE.is_file():
        raise FileNotFoundError(
            "V2 threshold analysis not found: "
            f"{THRESHOLD_ANALYSIS_FILE}"
        )

    # -------------------------------------------------
    # Freeze validation-selected threshold
    # -------------------------------------------------

    threshold_analysis = json.loads(
        THRESHOLD_ANALYSIS_FILE.read_text(
            encoding="utf-8"
        )
    )

    threshold = float(
        threshold_analysis[
            "youden"
        ][
            "threshold"
        ]
    )

    print(
        "Selected model:      "
        "baseline_cnn_v2"
    )

    print(
        f"Frozen threshold:    "
        f"{threshold:.6f}"
    )

    print(
        "Threshold source:    "
        "validation_youden"
    )

    print(
        f"Source checkpoint:   "
        f"{SOURCE_MODEL}"
    )

    # -------------------------------------------------
    # Freeze final V2 model artifact
    # -------------------------------------------------

    FINAL_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        SOURCE_MODEL,
        FINAL_MODEL,
    )

    print(
        f"Frozen model:        "
        f"{FINAL_MODEL}"
    )

    # -------------------------------------------------
    # Load model and cached V2 test data
    # -------------------------------------------------

    model = tf.keras.models.load_model(
        FINAL_MODEL,
        compile=False,
    )

    dataset = build_cached_dataset_v2(
        split_name="test",
        batch_size=16,
        shuffle=False,
    )

    y_true_batches = []
    probability_batches = []

    print(
        "\nRunning final V2 test inference..."
    )

    for images, labels in dataset:
        probabilities = model(
            images,
            training=False,
        )

        y_true_batches.append(
            labels.numpy().reshape(-1)
        )

        probability_batches.append(
            probabilities.numpy().reshape(-1)
        )

    y_true = np.concatenate(
        y_true_batches
    ).astype(int)

    probabilities = np.concatenate(
        probability_batches
    )

    # -------------------------------------------------
    # Integrity checks
    # -------------------------------------------------

    if len(y_true) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Unexpected test sample count: "
            f"{len(y_true)}. "
            f"Expected {EXPECTED_TEST_SAMPLES}."
        )

    if len(probabilities) != len(y_true):
        raise RuntimeError(
            "Probability count does not match "
            "label count."
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            "Non-finite model probabilities "
            "were detected."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise RuntimeError(
            "Probabilities outside [0, 1] "
            "were detected."
        )

    print(
        f"Test samples:        "
        f"{len(y_true)}"
    )

    print(
        "Probability range:   "
        f"{probabilities.min():.6f} "
        f"to "
        f"{probabilities.max():.6f}"
    )

    # -------------------------------------------------
    # Apply frozen validation threshold
    # -------------------------------------------------

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # -------------------------------------------------
    # Final metrics
    # -------------------------------------------------

    metrics = {
        "evaluation_type": (
            "final_v2_held_out_test"
        ),
        "timestamp": (
            datetime.now().isoformat()
        ),
        "selected_model": (
            "baseline_cnn_v2"
        ),
        "model_file": str(
            FINAL_MODEL
        ),
        "source_checkpoint": str(
            SOURCE_MODEL
        ),
        "threshold": (
            threshold
        ),
        "threshold_source": (
            "validation_youden"
        ),
        "data_pipeline": (
            "baseline_v2_cached"
        ),
        "samples": int(
            len(y_true)
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "sensitivity": float(
            recall_score(
                y_true,
                y_pred,
            )
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "true_label": (
                y_true
            ),
            "probability": (
                probabilities
            ),
            "predicted_label": (
                y_pred
            ),
        }
    ).to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # -------------------------------------------------
    # Report
    # -------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL V2 HELD-OUT TEST RESULTS"
    )
    print("=" * 70)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{name:24s}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{name:24s}: "
                f"{value}"
            )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NORMAL",
                "PNEUMONIA",
            ],
            digits=4,
        )
    )

    print(
        f"Final V2 model: "
        f"{FINAL_MODEL}"
    )

    print(
        f"Metrics:        "
        f"{METRICS_FILE}"
    )

    print(
        f"Predictions:    "
        f"{PREDICTIONS_FILE}"
    )

    print(
        "\nFINAL V2 TEST EVALUATION "
        "STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()