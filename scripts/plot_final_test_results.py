import json

import matplotlib.pyplot as plt
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
    METRICS_DIR / "final_test_predictions.csv"
)

METRICS_FILE = (
    METRICS_DIR / "final_test_metrics.json"
)


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    metrics = json.loads(
        METRICS_FILE.read_text()
    )

    y_true = predictions[
        "true_label"
    ].to_numpy(dtype=int)

    probabilities = predictions[
        "probability"
    ].to_numpy()

    y_pred = predictions[
        "predicted_label"
    ].to_numpy(dtype=int)

    threshold = metrics["threshold"]

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name="MediScan Final CNN",
    )

    plt.title(
        "MediScan — Held-Out Test ROC Curve"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "final_test_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Precision-recall curve
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name="MediScan Final CNN",
    )

    plt.title(
        "MediScan — Held-Out Test Precision-Recall Curve"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "final_test_pr_curve.png",
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
        "MediScan — Held-Out Test Confusion Matrix\n"
        f"Frozen Validation Threshold = {threshold:.4f}"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR
        / "final_test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("=" * 70)
    print("MEDISCAN AI — FINAL TEST FIGURES")
    print("=" * 70)

    print(f"Frozen threshold: {threshold:.4f}")

    print("\nGenerated:")

    for name in (
        "final_test_roc_curve.png",
        "final_test_pr_curve.png",
        "final_test_confusion_matrix.png",
    ):
        print(f"  {FIGURES_DIR / name}")

    print("\nFINAL TEST FIGURE STATUS: READY")


if __name__ == "__main__":
    main()