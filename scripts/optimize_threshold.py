import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from backend.app.core.config import PROJECT_ROOT


PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_validation_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_threshold_analysis.json"
)

MIN_SCREENING_SENSITIVITY = 0.95


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:

    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp)

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
        "specificity": float(specificity),
        "f1": float(
            f1_score(y_true, y_pred)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    df = pd.read_csv(PREDICTIONS_FILE)

    y_true = df["true_label"].to_numpy(dtype=int)
    probabilities = df["probability"].to_numpy()

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        probabilities,
    )

    youden_index = tpr - fpr
    youden_threshold = roc_thresholds[
        np.argmax(youden_index)
    ]

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                probabilities,
                np.linspace(0.0, 1.0, 1001),
            ]
        )
    )

    candidate_results = [
        calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )
        for threshold in candidate_thresholds
    ]

    best_f1 = max(
        candidate_results,
        key=lambda result: result["f1"],
    )

    screening_candidates = [
        result
        for result in candidate_results
        if result["sensitivity"]
        >= MIN_SCREENING_SENSITIVITY
    ]

    screening = max(
        screening_candidates,
        key=lambda result: (
            result["specificity"],
            result["precision"],
            result["threshold"],
        ),
    )

    results = {
        "default_0_5": calculate_metrics(
            y_true,
            probabilities,
            0.5,
        ),
        "youden": calculate_metrics(
            y_true,
            probabilities,
            youden_threshold,
        ),
        "best_f1": best_f1,
        "screening_95_sensitivity": screening,
    }

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2)
    )

    print("=" * 70)
    print("MEDISCAN AI — THRESHOLD OPTIMIZATION")
    print("=" * 70)

    for strategy, metrics in results.items():
        print(f"\n{strategy.upper()}")

        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name:14s}: {value:.4f}")
            else:
                print(f"  {name:14s}: {value}")

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    