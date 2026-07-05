import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from backend.app.core.config import PROJECT_ROOT


METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

PREDICTIONS_FILE = (
    METRICS_DIR / "baseline_validation_predictions.csv"
)

THRESHOLD_FILE = (
    METRICS_DIR / "baseline_threshold_analysis.json"
)


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(PREDICTIONS_FILE)

    threshold_results = json.loads(
        THRESHOLD_FILE.read_text()
    )

    y_true = predictions[
        "true_label"
    ].to_numpy(dtype=int)

    probabilities = predictions[
        "probability"
    ].to_numpy()

    threshold = threshold_results[
        "youden"
    ]["threshold"]

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name="Baseline CNN",
    )

    plt.title(
        "Baseline CNN — Validation ROC Curve"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "baseline_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Precision-recall curve
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name="Baseline CNN",
    )

    plt.title(
        "Baseline CNN — Validation Precision-Recall Curve"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "baseline_pr_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Confusion matrix
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "NORMAL",
            "PNEUMONIA",
        ],
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        f"Baseline CNN — Validation Confusion Matrix\n"
        f"Threshold = {threshold:.4f}"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR
        / "baseline_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("=" * 70)
    print("MEDISCAN AI — BASELINE DIAGNOSTIC FIGURES")
    print("=" * 70)
    print(f"Operating threshold: {threshold:.4f}")
    print("\nGenerated:")

    for name in (
        "baseline_roc_curve.png",
        "baseline_pr_curve.png",
        "baseline_confusion_matrix.png",
    ):
        print(f"  {FIGURES_DIR / name}")

    print("\nBASELINE FIGURE STATUS: READY")


if __name__ == "__main__":
    main()