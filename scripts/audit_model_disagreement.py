from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

from backend.app.core.config import PROJECT_ROOT


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

V1_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)

V2_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "final_test_v2_predictions.csv"
)

V1_METRICS_PATH = (
    RESULTS_DIR
    / "final_test_metrics.json"
)

V2_METRICS_PATH = (
    RESULTS_DIR
    / "final_test_v2_metrics.json"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "model_disagreement_audit.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "model_disagreement_audit.json"
)


def safe_rate(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def calculate_group_metrics(
    group: pd.DataFrame,
) -> dict:

    samples = len(group)

    if samples == 0:
        return {
            "samples": 0,
            "v1_accuracy": None,
            "v2_accuracy": None,
            "v1_errors": 0,
            "v2_errors": 0,
            "v1_false_negatives": 0,
            "v1_false_positives": 0,
        }

    return {
        "samples": int(samples),

        "v1_accuracy": float(
            accuracy_score(
                group["true_label"],
                group["v1_predicted_label"],
            )
        ),

        "v2_accuracy": float(
            accuracy_score(
                group["true_label"],
                group["v2_predicted_label"],
            )
        ),

        "v1_errors": int(
            (
                group["v1_correct"] == 0
            ).sum()
        ),

        "v2_errors": int(
            (
                group["v2_correct"] == 0
            ).sum()
        ),

        "v1_false_negatives": int(
            (
                (
                    group["true_label"] == 1
                )
                &
                (
                    group[
                        "v1_predicted_label"
                    ] == 0
                )
            ).sum()
        ),

        "v1_false_positives": int(
            (
                (
                    group["true_label"] == 0
                )
                &
                (
                    group[
                        "v1_predicted_label"
                    ] == 1
                )
            ).sum()
        ),
    }


def main() -> None:

    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "V1/V2 DISAGREEMENT AND TRIAGE AUDIT"
    )
    print("=" * 70)

    print(
        "\nLoading saved test predictions..."
    )

    v1 = pd.read_csv(
        V1_PREDICTIONS_PATH
    )

    v2 = pd.read_csv(
        V2_PREDICTIONS_PATH
    )

    print(
        f"V1 rows: {len(v1):,}"
    )

    print(
        f"V2 rows: {len(v2):,}"
    )

    if len(v1) != len(v2):
        raise RuntimeError(
            "V1/V2 sample counts do not match."
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
        "Label order: "
        f"{'PASS' if labels_match else 'FAIL'}"
    )

    if not labels_match:
        raise RuntimeError(
            "V1/V2 test labels are not aligned."
        )

    v1_metrics = json.loads(
        V1_METRICS_PATH.read_text()
    )

    v2_metrics = json.loads(
        V2_METRICS_PATH.read_text()
    )

    v1_threshold = float(
        v1_metrics["threshold"]
    )

    v2_threshold = float(
        v2_metrics["threshold"]
    )

    print(
        f"V1 threshold: {v1_threshold:.6f}"
    )

    print(
        f"V2 threshold: {v2_threshold:.6f}"
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

    v1_prediction = (
        v1_probability >= v1_threshold
    ).astype(np.int32)

    v2_prediction = (
        v2_probability >= v2_threshold
    ).astype(np.int32)

    audit = pd.DataFrame(
        {
            "true_label": y_true,
            "v1_probability": v1_probability,
            "v2_probability": v2_probability,
            "v1_predicted_label": v1_prediction,
            "v2_predicted_label": v2_prediction,
        }
    )

    audit["models_agree"] = (
        audit["v1_predicted_label"]
        ==
        audit["v2_predicted_label"]
    ).astype(np.int32)

    audit["v1_correct"] = (
        audit["v1_predicted_label"]
        ==
        audit["true_label"]
    ).astype(np.int32)

    audit["v2_correct"] = (
        audit["v2_predicted_label"]
        ==
        audit["true_label"]
    ).astype(np.int32)

    audit["probability_gap"] = np.abs(
        audit["v1_probability"]
        -
        audit["v2_probability"]
    )

    audit["v1_threshold_distance"] = np.abs(
        audit["v1_probability"]
        -
        v1_threshold
    )

    audit["v2_threshold_distance"] = np.abs(
        audit["v2_probability"]
        -
        v2_threshold
    )

    audit["minimum_threshold_distance"] = np.minimum(
        audit["v1_threshold_distance"],
        audit["v2_threshold_distance"],
    )

    audit["case_type"] = np.select(
        [
            (
                (audit["v1_correct"] == 1)
                &
                (audit["v2_correct"] == 1)
            ),

            (
                (audit["v1_correct"] == 1)
                &
                (audit["v2_correct"] == 0)
            ),

            (
                (audit["v1_correct"] == 0)
                &
                (audit["v2_correct"] == 1)
            ),
        ],
        [
            "both_correct",
            "only_v1_correct",
            "only_v2_correct",
        ],
        default="both_wrong",
    )

    agree = audit[
        audit["models_agree"] == 1
    ]

    disagree = audit[
        audit["models_agree"] == 0
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "DISAGREEMENT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total samples:       {len(audit)}"
    )

    print(
        f"Models agree:        {len(agree)}"
    )

    print(
        f"Models disagree:     {len(disagree)}"
    )

    print(
        "Disagreement rate:   "
        f"{len(disagree) / len(audit):.4f}"
    )

    case_counts = (
        audit["case_type"]
        .value_counts()
        .reindex(
            [
                "both_correct",
                "only_v1_correct",
                "only_v2_correct",
                "both_wrong",
            ],
            fill_value=0,
        )
    )

    print(
        "\nComplementarity:"
    )

    for case_type, count in (
        case_counts.items()
    ):
        print(
            f"{case_type:18s}: {count}"
        )

    agree_metrics = (
        calculate_group_metrics(
            agree
        )
    )

    disagree_metrics = (
        calculate_group_metrics(
            disagree
        )
    )

    print(
        "\nV1 performance when models agree:"
    )

    print(
        f"  Samples:   "
        f"{agree_metrics['samples']}"
    )

    print(
        f"  Accuracy:  "
        f"{agree_metrics['v1_accuracy']:.4f}"
    )

    print(
        f"  Errors:    "
        f"{agree_metrics['v1_errors']}"
    )

    print(
        "\nV1 performance when models disagree:"
    )

    print(
        f"  Samples:   "
        f"{disagree_metrics['samples']}"
    )

    print(
        f"  Accuracy:  "
        f"{disagree_metrics['v1_accuracy']:.4f}"
    )

    print(
        f"  Errors:    "
        f"{disagree_metrics['v1_errors']}"
    )

    total_v1_errors = int(
        (
            audit["v1_correct"] == 0
        ).sum()
    )

    v1_errors_in_disagreement = int(
        (
            disagree["v1_correct"] == 0
        ).sum()
    )

    error_capture_rate = safe_rate(
        v1_errors_in_disagreement,
        total_v1_errors,
    )

    print(
        "\nV1 error capture by disagreement:"
    )

    print(
        f"  Total V1 errors:          "
        f"{total_v1_errors}"
    )

    print(
        f"  Errors in disagreement:   "
        f"{v1_errors_in_disagreement}"
    )

    print(
        f"  Error capture rate:       "
        f"{error_capture_rate:.4f}"
    )

    v1_false_negatives = audit[
        (audit["true_label"] == 1)
        &
        (audit["v1_predicted_label"] == 0)
    ]

    fn_captured = int(
        (
            v1_false_negatives[
                "models_agree"
            ] == 0
        ).sum()
    )

    fn_capture_rate = safe_rate(
        fn_captured,
        len(v1_false_negatives),
    )

    print(
        "\nV1 false-negative capture:"
    )

    print(
        f"  Total V1 false negatives: "
        f"{len(v1_false_negatives)}"
    )

    print(
        f"  Caught by disagreement:   "
        f"{fn_captured}"
    )

    print(
        f"  FN capture rate:          "
        f"{fn_capture_rate:.4f}"
    )

    summary = {
        "evaluation_type": (
            "v1_v2_disagreement_triage_audit"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "samples": int(
            len(audit)
        ),

        "v1_threshold": v1_threshold,

        "v2_threshold": v2_threshold,

        "agreement": {
            "samples": int(
                len(agree)
            ),

            "rate": float(
                len(agree) / len(audit)
            ),

            **agree_metrics,
        },

        "disagreement": {
            "samples": int(
                len(disagree)
            ),

            "rate": float(
                len(disagree) / len(audit)
            ),

            **disagree_metrics,
        },

        "complementarity": {
            key: int(value)
            for key, value
            in case_counts.items()
        },

        "v1_error_capture": {
            "total_v1_errors": (
                total_v1_errors
            ),

            "errors_in_disagreement": (
                v1_errors_in_disagreement
            ),

            "capture_rate": (
                error_capture_rate
            ),
        },

        "v1_false_negative_capture": {
            "total_false_negatives": int(
                len(v1_false_negatives)
            ),

            "caught_by_disagreement": (
                fn_captured
            ),

            "capture_rate": (
                fn_capture_rate
            ),
        },

        "mean_probability_gap": {
            "agreement": float(
                agree[
                    "probability_gap"
                ].mean()
            ),

            "disagreement": float(
                disagree[
                    "probability_gap"
                ].mean()
            ),
        },
    }

    audit.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"\nCSV:  {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print(
        "\nV1/V2 DISAGREEMENT AND "
        "TRIAGE AUDIT STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()