import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.data_pipeline import build_dataset


MODEL_FILE = PROJECT_ROOT / "models" / "baseline_cnn_best.keras"

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_FILE = RESULTS_DIR / "baseline_validation_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "baseline_validation_predictions.csv"

THRESHOLD = 0.5


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — BASELINE VALIDATION EVALUATION")
    print("=" * 70)

    configure_training_runtime()

    dataset = build_dataset(
        split="val",
        batch_size=16,
        shuffle=False,
    )

    print(f"Loading checkpoint: {MODEL_FILE}")

    model = tf.keras.models.load_model(MODEL_FILE)

    y_true_batches = []
    y_probability_batches = []

    print("Running inference...")

    for images, labels in dataset:
        probabilities = model(
            images,
            training=False,
        )

        y_true_batches.append(
            labels.numpy().reshape(-1)
        )

        y_probability_batches.append(
            probabilities.numpy().reshape(-1)
        )

    y_true = np.concatenate(y_true_batches).astype(int)
    y_probability = np.concatenate(y_probability_batches)

    y_pred = (
        y_probability >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp)

    metrics = {
        "threshold": THRESHOLD,
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred)
        ),
        "recall_sensitivity": float(
            recall_score(y_true, y_pred)
        ),
        "specificity": float(specificity),
        "roc_auc": float(
            roc_auc_score(y_true, y_probability)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
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
        json.dumps(metrics, indent=2)
    )

    pd.DataFrame(
        {
            "true_label": y_true,
            "probability": y_probability,
            "predicted_label": y_pred,
        }
    ).to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS — THRESHOLD 0.50")
    print("=" * 70)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:24s}: {value:.4f}")
        else:
            print(f"{name:24s}: {value}")

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
        )
    )

    print(f"Metrics:     {METRICS_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()