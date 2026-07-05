from __future__ import annotations

import json

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


MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "advanced_v3_best.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

METRICS_FILE = (
    RESULTS_DIR
    / "advanced_v3_validation_metrics.json"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "advanced_v3_validation_predictions.csv"
)

THRESHOLD = 0.5
BATCH_SIZE = 16


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "ADVANCED MODEL V3 VALIDATION EVALUATION"
    )
    print("=" * 70)

    configure_training_runtime()

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"V3 checkpoint not found:\n{MODEL_FILE}"
        )

    print(f"\nLoading checkpoint:\n{MODEL_FILE}")

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )

    print("\nBuilding validation dataset...")

    dataset = build_cached_dataset_v2(
        split_name="val",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    y_true_batches = []
    probability_batches = []

    print("Running validation inference...")

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
    ).astype(np.int32)

    probabilities = np.concatenate(
        probability_batches
    ).astype(np.float64)

    if len(y_true) != len(probabilities):
        raise RuntimeError(
            "Label and probability counts do not match."
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            "Non-finite probabilities detected."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise RuntimeError(
            "Probabilities outside [0, 1] detected."
        )

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(np.int32)

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
            "advanced_v3_validation"
        ),
        "model": "advanced_v3",
        "model_file": str(MODEL_FILE),
        "data_pipeline": (
            "baseline_v2_cached"
        ),
        "threshold": THRESHOLD,
        "samples": int(len(y_true)),
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
        "recall_sensitivity": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    pd.DataFrame(
        {
            "true_label": y_true,
            "probability": probabilities,
            "predicted_label": y_pred,
        }
    ).to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print(
        "ADVANCED V3 VALIDATION RESULTS "
        "— THRESHOLD 0.50"
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

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NORMAL",
                "PNEUMONIA",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print(f"Metrics:     {METRICS_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")

    print(
        "\nADVANCED V3 VALIDATION "
        "EVALUATION STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()