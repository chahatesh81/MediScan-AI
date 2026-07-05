from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

POLICY_PATH = (
    RESULTS_DIR / "triage_policy_optimization.json"
)

V1_PREDICTIONS_PATH = (
    RESULTS_DIR / "final_test_predictions.csv"
)

V2_PREDICTIONS_PATH = (
    RESULTS_DIR / "final_test_v2_predictions.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR / "final_triage_metrics.json"
)

OUTPUT_CSV = (
    RESULTS_DIR / "final_triage_predictions.csv"
)


def safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def normalized_margin(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:

    return np.where(
        probabilities >= threshold,

        (
            probabilities - threshold
        )
        /
        max(
            1.0 - threshold,
            1e-8,
        ),

        (
            threshold - probabilities
        )
        /
        max(
            threshold,
            1e-8,
        ),
    )


def main() -> None:

    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "FINAL FROZEN TRIAGE BENCHMARK"
    )
    print("=" * 70)

    print(
        "\nLoading frozen validation policy..."
    )

    policy_data = json.loads(
        POLICY_PATH.read_text()
    )

    if (
        policy_data["selection_source"]
        != "validation_only"
    ):
        raise RuntimeError(
            "Triage policy was not selected "
            "using validation-only data."
        )

    selected = policy_data[
        "selected_policy"
    ]

    v1_threshold = float(
        policy_data["v1_threshold"]
    )

    v2_threshold = float(
        policy_data["v2_threshold"]
    )

    v1_margin_cutoff = float(
        selected["v1_margin_cutoff"]
    )

    v2_margin_cutoff = float(
        selected["v2_margin_cutoff"]
    )

    print(
        f"V1 threshold:       "
        f"{v1_threshold:.6f}"
    )

    print(
        f"V2 threshold:       "
        f"{v2_threshold:.6f}"
    )

    print(
        f"V1 margin cutoff:   "
        f"{v1_margin_cutoff:.4f}"
    )

    print(
        f"V2 margin cutoff:   "
        f"{v2_margin_cutoff:.4f}"
    )

    print(
        "\nLoading saved held-out test predictions..."
    )

    v1 = pd.read_csv(
        V1_PREDICTIONS_PATH
    )

    v2 = pd.read_csv(
        V2_PREDICTIONS_PATH
    )

    print(
        f"V1 test rows: {len(v1):,}"
    )

    print(
        f"V2 test rows: {len(v2):,}"
    )

    if len(v1) != len(v2):
        raise RuntimeError(
            "V1/V2 test sample counts "
            "do not match."
        )

    y_true_v1 = (
        v1["true_label"]
        .to_numpy(dtype=np.int32)
    )

    y_true_v2 = (
        v2["true_label"]
        .to_numpy(dtype=np.int32)
    )

    labels_match = np.array_equal(
        y_true_v1,
        y_true_v2,
    )

    print(
        "Test label order: "
        f"{'PASS' if labels_match else 'FAIL'}"
    )

    if not labels_match:
        raise RuntimeError(
            "V1/V2 test labels "
            "are not aligned."
        )

    y_true = y_true_v1

    v1_probability = (
        v1["probability"]
        .to_numpy(dtype=np.float64)
    )

    v2_probability = (
        v2["probability"]
        .to_numpy(dtype=np.float64)
    )

    if not (
        np.isfinite(v1_probability).all()
        and
        np.isfinite(v2_probability).all()
    ):
        raise RuntimeError(
            "Non-finite probabilities found."
        )

    v1_prediction = (
        v1_probability >= v1_threshold
    ).astype(np.int32)

    v2_prediction = (
        v2_probability >= v2_threshold
    ).astype(np.int32)

    models_disagree = (
        v1_prediction != v2_prediction
    )

    v1_margin = normalized_margin(
        probabilities=v1_probability,
        threshold=v1_threshold,
    )

    v2_margin = normalized_margin(
        probabilities=v2_probability,
        threshold=v2_threshold,
    )

    review_mask = (
        models_disagree
        |
        (
            v1_margin
            <= v1_margin_cutoff
        )
        |
        (
            v2_margin
            <= v2_margin_cutoff
        )
    )

    auto_mask = ~review_mask

    v1_error = (
        v1_prediction != y_true
    )

    v1_false_negative = (
        (y_true == 1)
        &
        (v1_prediction == 0)
    )

    v1_false_positive = (
        (y_true == 0)
        &
        (v1_prediction == 1)
    )

    samples = len(y_true)

    review_count = int(
        review_mask.sum()
    )

    auto_count = int(
        auto_mask.sum()
    )

    total_v1_errors = int(
        v1_error.sum()
    )

    errors_captured = int(
        (
            v1_error
            &
            review_mask
        ).sum()
    )

    total_false_negatives = int(
        v1_false_negative.sum()
    )

    false_negatives_captured = int(
        (
            v1_false_negative
            &
            review_mask
        ).sum()
    )

    total_false_positives = int(
        v1_false_positive.sum()
    )

    false_positives_captured = int(
        (
            v1_false_positive
            &
            review_mask
        ).sum()
    )

    auto_errors = int(
        (
            v1_error
            &
            auto_mask
        ).sum()
    )

    auto_false_negatives = int(
        (
            v1_false_negative
            &
            auto_mask
        ).sum()
    )

    auto_false_positives = int(
        (
            v1_false_positive
            &
            auto_mask
        ).sum()
    )

    if auto_count == 0:
        raise RuntimeError(
            "Frozen policy produced "
            "zero automatic cases."
        )

    auto_accuracy = float(
        (
            v1_prediction[auto_mask]
            ==
            y_true[auto_mask]
        ).mean()
    )

    triage_action = np.where(
        review_mask,
        "REVIEW_REQUIRED",
        np.where(
            v1_prediction == 1,
            "AUTO_PNEUMONIA",
            "AUTO_NORMAL",
        ),
    )

    output_table = pd.DataFrame(
        {
            "true_label": y_true,
            "v1_probability": v1_probability,
            "v2_probability": v2_probability,
            "v1_predicted_label": v1_prediction,
            "v2_predicted_label": v2_prediction,
            "models_disagree": models_disagree,
            "v1_normalized_margin": v1_margin,
            "v2_normalized_margin": v2_margin,
            "triage_action": triage_action,
            "v1_correct": (
                v1_prediction == y_true
            ).astype(np.int32),
        }
    )

    metrics = {
        "evaluation_type": (
            "final_frozen_triage_benchmark"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "policy_selection_source": (
            "validation_only"
        ),

        "automatic_decision_source": (
            "v1_primary_classifier"
        ),

        "v1_threshold": v1_threshold,
        "v2_threshold": v2_threshold,

        "v1_margin_cutoff": (
            v1_margin_cutoff
        ),

        "v2_margin_cutoff": (
            v2_margin_cutoff
        ),

        "samples": int(samples),

        "review_count": review_count,

        "review_rate": safe_rate(
            review_count,
            samples,
        ),

        "auto_count": auto_count,

        "coverage": safe_rate(
            auto_count,
            samples,
        ),

        "auto_accuracy": (
            auto_accuracy
        ),

        "total_v1_errors": (
            total_v1_errors
        ),

        "errors_captured_for_review": (
            errors_captured
        ),

        "error_capture_rate": safe_rate(
            errors_captured,
            total_v1_errors,
        ),

        "total_v1_false_negatives": (
            total_false_negatives
        ),

        "false_negatives_captured": (
            false_negatives_captured
        ),

        "false_negative_capture_rate": (
            safe_rate(
                false_negatives_captured,
                total_false_negatives,
            )
        ),

        "total_v1_false_positives": (
            total_false_positives
        ),

        "false_positives_captured": (
            false_positives_captured
        ),

        "false_positive_capture_rate": (
            safe_rate(
                false_positives_captured,
                total_false_positives,
            )
        ),

        "auto_errors_remaining": (
            auto_errors
        ),

        "auto_false_negatives_remaining": (
            auto_false_negatives
        ),

        "auto_false_positives_remaining": (
            auto_false_positives
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    output_table.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TRIAGE RESULTS"
    )

    print(
        "=" * 70
    )

    for name, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{name:32s}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{name:32s}: "
                f"{value}"
            )

    print(
        f"\nMetrics:     {OUTPUT_JSON}"
    )

    print(
        f"Predictions: {OUTPUT_CSV}"
    )

    print(
        "\nFINAL FROZEN TRIAGE "
        "BENCHMARK STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()