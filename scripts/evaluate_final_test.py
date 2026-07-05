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
from backend.app.services.data_pipeline import build_dataset


SOURCE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "baseline_cnn_best.keras"
)

FINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mediscan_final.keras"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "model_comparison.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

METRICS_FILE = (
    RESULTS_DIR
    / "final_test_metrics.json"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — FINAL HELD-OUT TEST EVALUATION")
    print("=" * 70)

    configure_training_runtime()

    comparison = json.loads(
        COMPARISON_FILE.read_text()
    )

    selected_model = comparison["selected_model"]

    if selected_model != "baseline_cnn":
        raise RuntimeError(
            f"Expected baseline_cnn, got {selected_model}"
        )

    threshold = float(
        comparison["baseline_cnn"]["youden"]["threshold"]
    )

    print(f"Selected model:      {selected_model}")
    print(f"Frozen threshold:    {threshold:.6f}")
    print(f"Source checkpoint:   {SOURCE_MODEL}")

    shutil.copy2(
        SOURCE_MODEL,
        FINAL_MODEL,
    )

    model = tf.keras.models.load_model(
        FINAL_MODEL,
        compile=False,
    )

    dataset = build_dataset(
        split="test",
        batch_size=16,
        shuffle=False,
    )

    y_true_batches = []
    probability_batches = []

    print("\nRunning one-time held-out test inference...")

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

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp)

    metrics = {
        "evaluation_type": "final_held_out_test",
        "timestamp": datetime.now().isoformat(),
        "selected_model": selected_model,
        "model_file": str(FINAL_MODEL),
        "threshold": threshold,
        "threshold_source": "validation_youden",
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred)
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(y_true, y_pred)
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
        json.dumps(metrics, indent=2)
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
    print("FINAL HELD-OUT TEST RESULTS")
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

    print(f"Final model:  {FINAL_MODEL}")
    print(f"Metrics:      {METRICS_FILE}")
    print(f"Predictions:  {PREDICTIONS_FILE}")

    print("\nFINAL TEST EVALUATION STATUS: COMPLETE")


if __name__ == "__main__":
    main()