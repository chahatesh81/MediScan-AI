import json
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
from backend.app.services.data_pipeline import build_dataset
from backend.app.services.data_pipeline_v2_cached import (
    build_cached_dataset_v2,
)


V1_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "baseline_cnn_best.keras"
)

V2_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "baseline_cnn_v2_best.keras"
)

OPTIMIZATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "ensemble_v1_v2_optimization.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

METRICS_FILE = (
    RESULTS_DIR
    / "final_ensemble_metrics.json"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "final_ensemble_predictions.csv"
)

EXPECTED_TEST_SAMPLES = 618
BATCH_SIZE = 16


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:

    labels_batches = []
    probability_batches = []

    print(
        f"\nRunning inference: {model_name}"
    )

    for images, labels in dataset:
        probabilities = model(
            images,
            training=False,
        )

        labels_batches.append(
            labels.numpy().reshape(-1)
        )

        probability_batches.append(
            probabilities.numpy().reshape(-1)
        )

    labels = np.concatenate(
        labels_batches
    ).astype(np.int32)

    probabilities = np.concatenate(
        probability_batches
    ).astype(np.float64)

    print(
        f"Samples:             "
        f"{len(labels)}"
    )

    print(
        f"Probability range:   "
        f"{probabilities.min():.6f} "
        f"to "
        f"{probabilities.max():.6f}"
    )

    return labels, probabilities


def validate_probabilities(
    probabilities: np.ndarray,
    name: str,
) -> None:

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            f"{name} contains "
            "non-finite probabilities."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise RuntimeError(
            f"{name} contains probabilities "
            "outside [0, 1]."
        )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "FINAL V1/V2 ENSEMBLE BENCHMARK"
    )
    print("=" * 70)

    configure_training_runtime()

    # -------------------------------------------------
    # Validate required artifacts
    # -------------------------------------------------

    required_files = [
        V1_MODEL_FILE,
        V2_MODEL_FILE,
        OPTIMIZATION_FILE,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required artifact missing: "
                f"{file_path}"
            )

    # -------------------------------------------------
    # Load frozen validation-selected configuration
    # -------------------------------------------------

    optimization = json.loads(
        OPTIMIZATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    selected = optimization[
        "best_youden"
    ]

    v1_weight = float(
        selected["v1_weight"]
    )

    v2_weight = float(
        selected["v2_weight"]
    )

    threshold = float(
        selected["threshold"]
    )

    if not np.isclose(
        v1_weight + v2_weight,
        1.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Ensemble weights do not sum to 1."
        )

    print()
    print(
        "Selection source:    "
        "validation_best_youden"
    )

    print(
        f"V1 weight:           "
        f"{v1_weight:.8f}"
    )

    print(
        f"V2 weight:           "
        f"{v2_weight:.8f}"
    )

    print(
        f"Frozen threshold:    "
        f"{threshold:.10f}"
    )

    print(
        f"V1 checkpoint:       "
        f"{V1_MODEL_FILE}"
    )

    print(
        f"V2 checkpoint:       "
        f"{V2_MODEL_FILE}"
    )

    # -------------------------------------------------
    # Load both frozen models
    # -------------------------------------------------

    print(
        "\nLoading V1 model..."
    )

    v1_model = tf.keras.models.load_model(
        V1_MODEL_FILE,
        compile=False,
    )

    print(
        "Loading V2 model..."
    )

    v2_model = tf.keras.models.load_model(
        V2_MODEL_FILE,
        compile=False,
    )

    # -------------------------------------------------
    # Build deterministic test pipelines
    # -------------------------------------------------

    v1_dataset = build_dataset(
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    v2_dataset = build_cached_dataset_v2(
        split_name="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # -------------------------------------------------
    # Independent inference
    # -------------------------------------------------

    v1_labels, v1_probabilities = (
        collect_predictions(
            model=v1_model,
            dataset=v1_dataset,
            model_name="V1 original pipeline",
        )
    )

    v2_labels, v2_probabilities = (
        collect_predictions(
            model=v2_model,
            dataset=v2_dataset,
            model_name="V2 cached pipeline",
        )
    )

    # -------------------------------------------------
    # Strict alignment and integrity checks
    # -------------------------------------------------

    if len(v1_labels) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Unexpected V1 test sample count: "
            f"{len(v1_labels)}"
        )

    if len(v2_labels) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Unexpected V2 test sample count: "
            f"{len(v2_labels)}"
        )

    if not np.array_equal(
        v1_labels,
        v2_labels,
    ):
        mismatch_indices = np.flatnonzero(
            v1_labels != v2_labels
        )

        print(
            "First label mismatch indices:",
            mismatch_indices[:10].tolist(),
        )

        raise RuntimeError(
            "V1 and V2 test rows are "
            "not aligned. Ensemble stopped."
        )

    print()
    print(
        "V1/V2 sample count:  PASS"
    )

    print(
        "V1/V2 label order:   PASS"
    )

    validate_probabilities(
        v1_probabilities,
        "V1",
    )

    validate_probabilities(
        v2_probabilities,
        "V2",
    )

    print(
        "Probability checks:  PASS"
    )

    # -------------------------------------------------
    # Frozen probability fusion
    # -------------------------------------------------

    ensemble_probabilities = (
        v1_weight
        * v1_probabilities
        + v2_weight
        * v2_probabilities
    )

    validate_probabilities(
        ensemble_probabilities,
        "Ensemble",
    )

    y_true = v1_labels

    y_pred = (
        ensemble_probabilities
        >= threshold
    ).astype(np.int32)

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

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

    metrics = {
        "evaluation_type": (
            "final_v1_v2_ensemble_benchmark"
        ),
        "timestamp": (
            datetime.now().isoformat()
        ),
        "selection_source": (
            "validation_best_youden"
        ),
        "v1_model_file": str(
            V1_MODEL_FILE
        ),
        "v2_model_file": str(
            V2_MODEL_FILE
        ),
        "v1_weight": (
            v1_weight
        ),
        "v2_weight": (
            v2_weight
        ),
        "threshold": (
            threshold
        ),
        "threshold_source": (
            "validation_ensemble_optimization"
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
                ensemble_probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                ensemble_probabilities,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # -------------------------------------------------
    # Save outputs
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
            "v1_probability": (
                v1_probabilities
            ),
            "v2_probability": (
                v2_probabilities
            ),
            "ensemble_probability": (
                ensemble_probabilities
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
    # Final report
    # -------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL V1/V2 ENSEMBLE RESULTS"
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
        f"Metrics:     "
        f"{METRICS_FILE}"
    )

    print(
        f"Predictions: "
        f"{PREDICTIONS_FILE}"
    )

    print()
    print(
        "FINAL ENSEMBLE BENCHMARK "
        "STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()