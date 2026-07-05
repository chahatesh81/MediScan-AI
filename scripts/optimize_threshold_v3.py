from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from backend.app.core.config import PROJECT_ROOT


PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "advanced_v3_validation_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "advanced_v3_threshold_analysis.json"
)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (
        probabilities >= threshold
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
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
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
    unique_thresholds = np.unique(
        probabilities
    )

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in unique_thresholds:
        result = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        youden = (
            result["sensitivity"]
            + result["specificity"]
            - 1.0
        )

        if youden > best_score:
            best_score = youden
            best_threshold = float(threshold)

    return best_threshold


def find_best_f1_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    unique_thresholds = np.unique(
        probabilities
    )

    best_threshold = 0.5
    best_f1 = -np.inf

    for threshold in unique_thresholds:
        result = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_threshold = float(threshold)

    return best_threshold


def find_screening_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_sensitivity: float = 0.99,
) -> float:
    unique_thresholds = np.sort(
        np.unique(probabilities)
    )

    feasible = []

    for threshold in unique_thresholds:
        result = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        if (
            result["sensitivity"]
            >= minimum_sensitivity
        ):
            feasible.append(result)

    if not feasible:
        raise RuntimeError(
            "No threshold satisfies the "
            "minimum sensitivity requirement."
        )

    best = max(
        feasible,
        key=lambda result: (
            result["specificity"],
            result["accuracy"],
            result["threshold"],
        ),
    )

    return float(best["threshold"])


def print_result(
    name: str,
    result: dict,
) -> None:
    print(f"\n{name}")

    for key in [
        "threshold",
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
    ]:
        print(
            f"  {key:14s}: "
            f"{result[key]:.4f}"
        )

    print(
        "  tn/fp/fn/tp   : "
        f"{result['tn']}/"
        f"{result['fp']}/"
        f"{result['fn']}/"
        f"{result['tp']}"
    )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "ADVANCED V3 THRESHOLD OPTIMIZATION"
    )
    print("=" * 70)

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    y_true = predictions[
        "true_label"
    ].to_numpy(dtype=np.int32)

    probabilities = predictions[
        "probability"
    ].to_numpy(dtype=np.float64)

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
            "Probabilities outside [0, 1]."
        )

    thresholds = {
        "default_0_5": 0.5,
        "youden": find_youden_threshold(
            y_true,
            probabilities,
        ),
        "best_f1": find_best_f1_threshold(
            y_true,
            probabilities,
        ),
        "screening_99_sensitivity": (
            find_screening_threshold(
                y_true,
                probabilities,
                minimum_sensitivity=0.99,
            )
        ),
    }

    results = {
        name: calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )
        for name, threshold
        in thresholds.items()
    }

    output = {
        "evaluation_type": (
            "advanced_v3_validation_threshold_optimization"
        ),
        "selection_source": "validation_only",
        "samples": int(len(y_true)),
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
        **results,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    for name, result in results.items():
        print_result(
            name.upper(),
            result,
        )

    print(f"\nSaved: {OUTPUT_FILE}")

    print(
        "\nADVANCED V3 THRESHOLD "
        "OPTIMIZATION STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()