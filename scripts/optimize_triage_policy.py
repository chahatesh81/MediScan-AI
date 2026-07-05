from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

V1_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "baseline_validation_predictions.csv"
)

V2_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "baseline_v2_validation_predictions.csv"
)

V1_METRICS_PATH = (
    RESULTS_DIR
    / "baseline_validation_metrics.json"
)

V2_THRESHOLD_ANALYSIS_PATH = (
    RESULTS_DIR
    / "baseline_v2_threshold_analysis.json"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "triage_policy_optimization.json"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "triage_policy_search.csv"
)

OUTPUT_PREDICTIONS_CSV = (
    RESULTS_DIR
    / "triage_policy_validation_predictions.csv"
)


MIN_AUTO_ACCURACY = 0.99
MAX_REVIEW_RATE = 0.25

V1_MARGIN_GRID = np.linspace(
    0.0,
    0.30,
    31,
)

V2_MARGIN_GRID = np.linspace(
    0.0,
    0.30,
    31,
)


def load_thresholds() -> tuple[float, float]:

    v1_comparison_path = (
        RESULTS_DIR
        / "model_comparison.json"
    )

    v2_analysis = json.loads(
        V2_THRESHOLD_ANALYSIS_PATH.read_text()
    )

    v1_comparison = json.loads(
        v1_comparison_path.read_text()
    )

    v1_threshold = float(
        v1_comparison[
            "baseline_cnn"
        ][
            "youden"
        ][
            "threshold"
        ]
    )

    v2_threshold = float(
        v2_analysis[
            "youden"
        ][
            "threshold"
        ]
    )

    return (
        v1_threshold,
        v2_threshold,
    )


def build_validation_table(
    v1_threshold: float,
    v2_threshold: float,
) -> pd.DataFrame:

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
            "V1/V2 validation sample counts "
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
        "Label order: "
        f"{'PASS' if labels_match else 'FAIL'}"
    )

    if not labels_match:
        raise RuntimeError(
            "V1/V2 validation labels "
            "are not aligned."
        )

    table = pd.DataFrame(
        {
            "true_label": y_true_v1,

            "v1_probability": (
                v1["probability"]
                .to_numpy(dtype=np.float64)
            ),

            "v2_probability": (
                v2["probability"]
                .to_numpy(dtype=np.float64)
            ),
        }
    )

    table["v1_predicted_label"] = (
        table["v1_probability"]
        >= v1_threshold
    ).astype(np.int32)

    table["v2_predicted_label"] = (
        table["v2_probability"]
        >= v2_threshold
    ).astype(np.int32)

    table["models_disagree"] = (
        table["v1_predicted_label"]
        != table["v2_predicted_label"]
    )

    table["v1_correct"] = (
        table["v1_predicted_label"]
        == table["true_label"]
    )

    table["v2_correct"] = (
        table["v2_predicted_label"]
        == table["true_label"]
    )

    # Normalize distance to each model's own
    # decision threshold.
    #
    # A value of 0 means exactly at threshold.
    # Larger values mean farther from threshold.

    table["v1_normalized_margin"] = np.where(
        table["v1_probability"]
        >= v1_threshold,

        (
            table["v1_probability"]
            - v1_threshold
        )
        /
        max(
            1.0 - v1_threshold,
            1e-8,
        ),

        (
            v1_threshold
            - table["v1_probability"]
        )
        /
        max(
            v1_threshold,
            1e-8,
        ),
    )

    table["v2_normalized_margin"] = np.where(
        table["v2_probability"]
        >= v2_threshold,

        (
            table["v2_probability"]
            - v2_threshold
        )
        /
        max(
            1.0 - v2_threshold,
            1e-8,
        ),

        (
            v2_threshold
            - table["v2_probability"]
        )
        /
        max(
            v2_threshold,
            1e-8,
        ),
    )

    return table


def evaluate_policy(
    table: pd.DataFrame,
    v1_margin_cutoff: float,
    v2_margin_cutoff: float,
) -> tuple[dict, np.ndarray]:

    disagreement_review = (
        table["models_disagree"]
        .to_numpy(dtype=bool)
    )

    v1_uncertain = (
        table["v1_normalized_margin"]
        .to_numpy()
        <= v1_margin_cutoff
    )

    v2_uncertain = (
        table["v2_normalized_margin"]
        .to_numpy()
        <= v2_margin_cutoff
    )

    review_mask = (
        disagreement_review
        |
        v1_uncertain
        |
        v2_uncertain
    )

    auto_mask = ~review_mask

    sample_count = len(table)

    review_count = int(
        review_mask.sum()
    )

    auto_count = int(
        auto_mask.sum()
    )

    review_rate = (
        review_count / sample_count
    )

    coverage = (
        auto_count / sample_count
    )

    y_true = (
        table["true_label"]
        .to_numpy(dtype=np.int32)
    )

    v1_prediction = (
        table["v1_predicted_label"]
        .to_numpy(dtype=np.int32)
    )

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

    total_errors = int(
        v1_error.sum()
    )

    total_false_negatives = int(
        v1_false_negative.sum()
    )

    total_false_positives = int(
        v1_false_positive.sum()
    )

    errors_captured = int(
        (
            v1_error
            &
            review_mask
        ).sum()
    )

    false_negatives_captured = int(
        (
            v1_false_negative
            &
            review_mask
        ).sum()
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

    if auto_count > 0:
        auto_accuracy = float(
            (
                v1_prediction[auto_mask]
                ==
                y_true[auto_mask]
            ).mean()
        )
    else:
        auto_accuracy = 0.0

    error_capture_rate = (
        errors_captured / total_errors
        if total_errors > 0
        else 0.0
    )

    fn_capture_rate = (
        false_negatives_captured
        / total_false_negatives
        if total_false_negatives > 0
        else 0.0
    )

    fp_capture_rate = (
        false_positives_captured
        / total_false_positives
        if total_false_positives > 0
        else 0.0
    )

    metrics = {
        "v1_margin_cutoff": float(
            v1_margin_cutoff
        ),

        "v2_margin_cutoff": float(
            v2_margin_cutoff
        ),

        "samples": int(
            sample_count
        ),

        "review_count": (
            review_count
        ),

        "review_rate": float(
            review_rate
        ),

        "auto_count": (
            auto_count
        ),

        "coverage": float(
            coverage
        ),

        "auto_accuracy": float(
            auto_accuracy
        ),

        "total_v1_errors": (
            total_errors
        ),

        "errors_captured_for_review": (
            errors_captured
        ),

        "error_capture_rate": float(
            error_capture_rate
        ),

        "total_v1_false_negatives": (
            total_false_negatives
        ),

        "false_negatives_captured": (
            false_negatives_captured
        ),

        "false_negative_capture_rate": float(
            fn_capture_rate
        ),

        "total_v1_false_positives": (
            total_false_positives
        ),

        "false_positives_captured": (
            false_positives_captured
        ),

        "false_positive_capture_rate": float(
            fp_capture_rate
        ),

        "auto_errors": (
            auto_errors
        ),

        "auto_false_negatives": (
            auto_false_negatives
        ),

        "auto_false_positives": (
            auto_false_positives
        ),
    }

    return (
        metrics,
        review_mask,
    )


def select_best_policy(
    records: list[dict],
) -> dict:

    feasible = [
        record
        for record in records
        if (
            record["auto_accuracy"]
            >= MIN_AUTO_ACCURACY
            and
            record["review_rate"]
            <= MAX_REVIEW_RATE
        )
    ]

    print(
        f"\nFeasible policies: "
        f"{len(feasible):,} / "
        f"{len(records):,}"
    )

    if feasible:

        # Priority:
        # 1. Catch false negatives.
        # 2. Catch total errors.
        # 3. Increase automatic accuracy.
        # 4. Reduce review burden.

        return max(
            feasible,
            key=lambda record: (
                record[
                    "false_negative_capture_rate"
                ],
                record[
                    "error_capture_rate"
                ],
                record[
                    "auto_accuracy"
                ],
                -record[
                    "review_rate"
                ],
            ),
        )

    print(
        "WARNING: No policy satisfied both "
        "hard constraints."
    )

    print(
        "Selecting best fallback policy."
    )

    return max(
        records,
        key=lambda record: (
            record[
                "false_negative_capture_rate"
            ],
            record[
                "error_capture_rate"
            ],
            record[
                "auto_accuracy"
            ],
            -record[
                "review_rate"
            ],
        ),
    )


def print_policy(
    policy: dict,
) -> None:

    print(
        f"V1 margin cutoff:       "
        f"{policy['v1_margin_cutoff']:.4f}"
    )

    print(
        f"V2 margin cutoff:       "
        f"{policy['v2_margin_cutoff']:.4f}"
    )

    print(
        f"Review count:           "
        f"{policy['review_count']}"
    )

    print(
        f"Review rate:            "
        f"{policy['review_rate']:.4f}"
    )

    print(
        f"Automatic count:        "
        f"{policy['auto_count']}"
    )

    print(
        f"Coverage:               "
        f"{policy['coverage']:.4f}"
    )

    print(
        f"Automatic accuracy:     "
        f"{policy['auto_accuracy']:.4f}"
    )

    print(
        f"Error capture rate:     "
        f"{policy['error_capture_rate']:.4f}"
    )

    print(
        f"FN capture rate:        "
        f"{policy['false_negative_capture_rate']:.4f}"
    )

    print(
        f"FP capture rate:        "
        f"{policy['false_positive_capture_rate']:.4f}"
    )

    print(
        f"Auto errors remaining:  "
        f"{policy['auto_errors']}"
    )

    print(
        f"Auto FN remaining:      "
        f"{policy['auto_false_negatives']}"
    )

    print(
        f"Auto FP remaining:      "
        f"{policy['auto_false_positives']}"
    )


def main() -> None:

    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "VALIDATION-ONLY TRIAGE POLICY OPTIMIZATION"
    )
    print("=" * 70)

    (
        v1_threshold,
        v2_threshold,
    ) = load_thresholds()

    print(
        f"\nV1 threshold: {v1_threshold:.6f}"
    )

    print(
        f"V2 threshold: {v2_threshold:.6f}"
    )

    print(
        "\nBuilding aligned validation table..."
    )

    table = build_validation_table(
        v1_threshold=v1_threshold,
        v2_threshold=v2_threshold,
    )

    print(
        f"Validation samples: {len(table):,}"
    )

    disagreement_count = int(
        table["models_disagree"].sum()
    )

    print(
        f"Validation disagreements: "
        f"{disagreement_count}"
    )

    print(
        "\nSearching triage policies..."
    )

    records = []

    total_searches = (
        len(V1_MARGIN_GRID)
        *
        len(V2_MARGIN_GRID)
    )

    print(
        f"Policies to evaluate: "
        f"{total_searches:,}"
    )

    for v1_margin_cutoff in (
        V1_MARGIN_GRID
    ):

        for v2_margin_cutoff in (
            V2_MARGIN_GRID
        ):

            metrics, _ = evaluate_policy(
                table=table,
                v1_margin_cutoff=float(
                    v1_margin_cutoff
                ),
                v2_margin_cutoff=float(
                    v2_margin_cutoff
                ),
            )

            records.append(
                metrics
            )

    selected = select_best_policy(
        records
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED VALIDATION TRIAGE POLICY"
    )

    print(
        "=" * 70
    )

    print_policy(
        selected
    )

    (
        _,
        selected_review_mask,
    ) = evaluate_policy(
        table=table,
        v1_margin_cutoff=(
            selected[
                "v1_margin_cutoff"
            ]
        ),
        v2_margin_cutoff=(
            selected[
                "v2_margin_cutoff"
            ]
        ),
    )

    table["triage_action"] = np.where(
        selected_review_mask,
        "REVIEW_REQUIRED",
        np.where(
            table["v1_predicted_label"]
            == 1,
            "AUTO_PNEUMONIA",
            "AUTO_NORMAL",
        ),
    )

    pd.DataFrame(
        records
    ).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    table.to_csv(
        OUTPUT_PREDICTIONS_CSV,
        index=False,
    )

    output = {
        "evaluation_type": (
            "validation_only_triage_"
            "policy_optimization"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "selection_source": (
            "validation_only"
        ),

        "policy_design": {
            "always_review_disagreement": (
                True
            ),

            "review_if_v1_margin_below": (
                selected[
                    "v1_margin_cutoff"
                ]
            ),

            "review_if_v2_margin_below": (
                selected[
                    "v2_margin_cutoff"
                ]
            ),

            "automatic_decision_source": (
                "v1_primary_classifier"
            ),
        },

        "constraints": {
            "minimum_auto_accuracy": (
                MIN_AUTO_ACCURACY
            ),

            "maximum_review_rate": (
                MAX_REVIEW_RATE
            ),
        },

        "v1_threshold": (
            v1_threshold
        ),

        "v2_threshold": (
            v2_threshold
        ),

        "selected_policy": (
            selected
        ),

        "policies_evaluated": int(
            len(records)
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print(
        f"\nJSON: {OUTPUT_JSON}"
    )

    print(
        f"Search CSV: {OUTPUT_CSV}"
    )

    print(
        "Validation predictions: "
        f"{OUTPUT_PREDICTIONS_CSV}"
    )

    print(
        "\nVALIDATION-ONLY TRIAGE POLICY "
        "OPTIMIZATION STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()