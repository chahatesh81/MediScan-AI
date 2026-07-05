import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_curve,
)

from backend.app.core.config import PROJECT_ROOT


PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_v2_validation_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_v2_threshold_analysis.json"
)


def calculate_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
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
            sensitivity
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "BASELINE V2 THRESHOLD OPTIMIZATION"
    )
    print("=" * 70)

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    y_true = predictions[
        "true_label"
    ].to_numpy(
        dtype=int
    )

    y_probability = predictions[
        "probability"
    ].to_numpy(
        dtype=float
    )

    if len(y_true) != 713:
        raise RuntimeError(
            "Unexpected validation sample count: "
            f"{len(y_true)}"
        )

    # -------------------------------------------------
    # Default threshold
    # -------------------------------------------------

    default_threshold = 0.5

    # -------------------------------------------------
    # Youden J threshold
    # -------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        y_probability,
    )

    finite_mask = np.isfinite(
        roc_thresholds
    )

    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    roc_thresholds = (
        roc_thresholds[finite_mask]
    )

    youden_j = (
        tpr - fpr
    )

    youden_index = int(
        np.argmax(youden_j)
    )

    youden_threshold = float(
        roc_thresholds[
            youden_index
        ]
    )

    # -------------------------------------------------
    # Best F1 threshold
    # -------------------------------------------------

    candidate_thresholds = np.unique(
        y_probability
    )

    best_f1_threshold = 0.5
    best_f1_value = -1.0

    for threshold in candidate_thresholds:
        y_pred = (
            y_probability >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if current_f1 > best_f1_value:
            best_f1_value = float(
                current_f1
            )

            best_f1_threshold = float(
                threshold
            )

    # -------------------------------------------------
    # Screening threshold:
    # highest threshold achieving >= 95% sensitivity
    # -------------------------------------------------

    screening_candidates = []

    for threshold in candidate_thresholds:
        result = calculate_metrics(
            y_true,
            y_probability,
            float(threshold),
        )

        if result["sensitivity"] >= 0.95:
            screening_candidates.append(
                result
            )

    if not screening_candidates:
        raise RuntimeError(
            "No threshold achieved "
            "95% sensitivity."
        )

    screening_result = max(
        screening_candidates,
        key=lambda item: (
            item["specificity"],
            item["precision"],
            item["threshold"],
        ),
    )

    screening_threshold = (
        screening_result[
            "threshold"
        ]
    )

    # -------------------------------------------------
    # Final analysis
    # -------------------------------------------------

    analysis = {
        "default_0_5": (
            calculate_metrics(
                y_true,
                y_probability,
                default_threshold,
            )
        ),
        "youden": (
            calculate_metrics(
                y_true,
                y_probability,
                youden_threshold,
            )
        ),
        "best_f1": (
            calculate_metrics(
                y_true,
                y_probability,
                best_f1_threshold,
            )
        ),
        "screening_95_sensitivity": (
            calculate_metrics(
                y_true,
                y_probability,
                screening_threshold,
            )
        ),
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            analysis,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    for name, metrics in analysis.items():
        print(name.upper())

        for key, value in metrics.items():
            if isinstance(value, float):
                print(
                    f"  {key:14s}: "
                    f"{value:.4f}"
                )
            else:
                print(
                    f"  {key:14s}: "
                    f"{value}"
                )

        print()

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()