import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.data_pipeline import build_dataset


MODELS = {
    "baseline_cnn": (
        PROJECT_ROOT
        / "models"
        / "baseline_cnn_best.keras"
    ),
    "mobilenet_stage_a": (
        PROJECT_ROOT
        / "models"
        / "mobilenet_v2_stage_a_best.keras"
    ),
    "mobilenet_stage_b": (
        PROJECT_ROOT
        / "models"
        / "mobilenet_v2_stage_b_best.keras"
    ),
}

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
OUTPUT_FILE = RESULTS_DIR / "model_comparison.json"
TABLE_FILE = RESULTS_DIR / "model_comparison.csv"


def threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
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
        "specificity": float(
            tn / (tn + fp)
        ),
        "f1": float(
            f1_score(y_true, y_pred)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_youden_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
    )

    index = np.argmax(tpr - fpr)

    return float(thresholds[index])


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — VALIDATION MODEL COMPARISON")
    print("=" * 70)

    configure_training_runtime()

    dataset = build_dataset(
        split="val",
        batch_size=16,
        shuffle=False,
    )

    y_true_batches = []

    for _, labels in dataset:
        y_true_batches.append(
            labels.numpy().reshape(-1)
        )

    y_true = np.concatenate(
        y_true_batches
    ).astype(int)

    all_results = {}
    table_rows = []

    for model_name, model_path in MODELS.items():
        print(f"\nEvaluating: {model_name}")
        print(f"Checkpoint: {model_path}")

        model = tf.keras.models.load_model(
            model_path
        )

        probability_batches = []

        for images, _ in dataset:
            probabilities = model(
                images,
                training=False,
            )

            probability_batches.append(
                probabilities.numpy().reshape(-1)
            )

        probabilities = np.concatenate(
            probability_batches
        )

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

        youden_threshold = find_youden_threshold(
            y_true,
            probabilities,
        )

        default_metrics = threshold_metrics(
            y_true,
            probabilities,
            0.5,
        )

        youden_metrics = threshold_metrics(
            y_true,
            probabilities,
            youden_threshold,
        )

        all_results[model_name] = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "default_0_5": default_metrics,
            "youden": youden_metrics,
        }

        table_rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "youden_threshold":
                    youden_threshold,
                "youden_accuracy":
                    youden_metrics["accuracy"],
                "youden_sensitivity":
                    youden_metrics["sensitivity"],
                "youden_specificity":
                    youden_metrics["specificity"],
                "youden_f1":
                    youden_metrics["f1"],
            }
        )

        print(f"ROC-AUC:       {roc_auc:.4f}")
        print(f"PR-AUC:        {pr_auc:.4f}")
        print(
            f"Youden threshold: "
            f"{youden_threshold:.4f}"
        )
        print(
            f"Sensitivity:    "
            f"{youden_metrics['sensitivity']:.4f}"
        )
        print(
            f"Specificity:    "
            f"{youden_metrics['specificity']:.4f}"
        )
        print(
            f"F1:             "
            f"{youden_metrics['f1']:.4f}"
        )

        del model
        tf.keras.backend.clear_session()

    comparison_df = pd.DataFrame(
        table_rows
    ).sort_values(
        by=["roc_auc", "pr_auc"],
        ascending=False,
    )

    winner = comparison_df.iloc[0]["model"]

    all_results["selected_model"] = winner

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            all_results,
            indent=2,
        )
    )

    comparison_df.to_csv(
        TABLE_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("MODEL RANKING")
    print("=" * 70)
    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(f"\nSelected model: {winner}")
    print(f"JSON: {OUTPUT_FILE}")
    print(f"CSV:  {TABLE_FILE}")


if __name__ == "__main__":
    main()