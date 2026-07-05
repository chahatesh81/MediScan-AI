import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from backend.app.core.config import PROJECT_ROOT


V1_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_validation_predictions.csv"
)

V2_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_v2_validation_predictions.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "ensemble_v1_v2_optimization.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "ensemble_v1_v2_search.csv"
)

WEIGHTS = np.linspace(
    0.0,
    1.0,
    101,
)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (
        probabilities >= threshold
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


def find_youden_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
    )

    finite = np.isfinite(
        thresholds
    )

    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]

    youden_j = tpr - fpr

    best_index = int(
        np.argmax(youden_j)
    )

    return float(
        thresholds[best_index]
    )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "V1/V2 ENSEMBLE OPTIMIZATION"
    )
    print("=" * 70)

    v1 = pd.read_csv(
        V1_FILE
    )

    v2 = pd.read_csv(
        V2_FILE
    )

    if len(v1) != 713 or len(v2) != 713:
        raise RuntimeError(
            "Expected 713 validation "
            "predictions per model."
        )

    y1 = v1[
        "true_label"
    ].to_numpy(
        dtype=int
    )

    y2 = v2[
        "true_label"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        y1,
        y2,
    ):
        raise RuntimeError(
            "V1 and V2 validation rows "
            "are not aligned."
        )

    y_true = y1

    p1 = v1[
        "probability"
    ].to_numpy(
        dtype=float
    )

    p2 = v2[
        "probability"
    ].to_numpy(
        dtype=float
    )

    records = []

    print(
        "\nSearching 101 fusion weights..."
    )

    for v1_weight in WEIGHTS:
        v2_weight = (
            1.0 - v1_weight
        )

        probabilities = (
            v1_weight * p1
            + v2_weight * p2
        )

        threshold = (
            find_youden_threshold(
                y_true,
                probabilities,
            )
        )

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        records.append(
            {
                "v1_weight": float(
                    v1_weight
                ),
                "v2_weight": float(
                    v2_weight
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
                **metrics,
            }
        )

    results = pd.DataFrame(
        records
    )

    best_youden = max(
        records,
        key=lambda row: (
            row["sensitivity"]
            + row["specificity"]
            - 1.0,
            row["roc_auc"],
            row["f1"],
        ),
    )

    best_f1 = max(
        records,
        key=lambda row: (
            row["f1"],
            row["sensitivity"],
            row["specificity"],
        ),
    )

    best_roc_auc = max(
        records,
        key=lambda row: (
            row["roc_auc"],
            row["pr_auc"],
        ),
    )

    # Reference configurations
    v1_only = records[-1]
    v2_only = records[0]

    analysis = {
        "selection_rule": (
            "validation_only_weight_and_threshold_search"
        ),
        "weights_tested": int(
            len(WEIGHTS)
        ),
        "v1_only": v1_only,
        "v2_only": v2_only,
        "best_youden": best_youden,
        "best_f1": best_f1,
        "best_roc_auc": best_roc_auc,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            analysis,
            indent=2,
        ),
        encoding="utf-8",
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()

    for name in [
        "v1_only",
        "v2_only",
        "best_youden",
        "best_f1",
        "best_roc_auc",
    ]:
        result = analysis[name]

        print(name.upper())

        print(
            f"  V1 weight:    "
            f"{result['v1_weight']:.2f}"
        )

        print(
            f"  V2 weight:    "
            f"{result['v2_weight']:.2f}"
        )

        print(
            f"  Threshold:    "
            f"{result['threshold']:.6f}"
        )

        print(
            f"  Accuracy:     "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"  Precision:    "
            f"{result['precision']:.4f}"
        )

        print(
            f"  Sensitivity:  "
            f"{result['sensitivity']:.4f}"
        )

        print(
            f"  Specificity:  "
            f"{result['specificity']:.4f}"
        )

        print(
            f"  F1:           "
            f"{result['f1']:.4f}"
        )

        print(
            f"  ROC-AUC:      "
            f"{result['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC:       "
            f"{result['pr_auc']:.4f}"
        )

        print(
            f"  TN/FP/FN/TP:  "
            f"{result['tn']}/"
            f"{result['fp']}/"
            f"{result['fn']}/"
            f"{result['tp']}"
        )

        print()

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print(
        f"CSV:  {OUTPUT_CSV}"
    )

    print(
        "\nENSEMBLE OPTIMIZATION "
        "STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()