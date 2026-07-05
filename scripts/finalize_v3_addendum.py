from __future__ import annotations

import json
from datetime import datetime

from backend.app.core.config import PROJECT_ROOT


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

V1_SELECTION_FILE = (
    RESULTS_DIR
    / "final_model_selection.json"
)

V3_TEST_METRICS_FILE = (
    RESULTS_DIR
    / "final_test_v3_metrics.json"
)

V1_V3_AUDIT_FILE = (
    RESULTS_DIR
    / "v1_v3_complementarity_audit.json"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "final_model_selection_v3_addendum.json"
)


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return json.loads(
        path.read_text()
    )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "V3 MODEL SELECTION ADDENDUM"
    )
    print("=" * 70)

    original_selection = load_json(
        V1_SELECTION_FILE
    )

    v3_metrics = load_json(
        V3_TEST_METRICS_FILE
    )

    complementarity = load_json(
        V1_V3_AUDIT_FILE
    )

    selected_primary_record = (
        original_selection.get(
            "selected_primary_model",
            original_selection.get(
                "selected_model",
                "baseline_cnn_v1",
            ),
        )
    )

    if isinstance(
        selected_primary_record,
        dict,
    ):
        selected_primary = (
            selected_primary_record.get(
                "name",
                "baseline_cnn_v1",
            )
        )
    else:
        selected_primary = str(
            selected_primary_record
        )
    v1_fn = int(
        complementarity[
            "v1_false_negatives"
        ]
    )

    v1_fn_caught_by_v3 = int(
        complementarity[
            "v1_false_negatives_caught_by_v3"
        ]
    )

    fn_rescue_rate = (
        v1_fn_caught_by_v3 / v1_fn
        if v1_fn > 0
        else 0.0
    )

    addendum = {
        "record_type": (
            "final_model_selection_v3_addendum"
        ),
        "timestamp": datetime.now().isoformat(),
        "original_decision_record": str(
            V1_SELECTION_FILE
        ),
        "primary_model_decision": {
            "model": selected_primary,
            "status": "UNCHANGED",
            "role": "primary_classifier",
        },
        "advanced_v3": {
            "model": "advanced_v3",
            "architecture": "EfficientNetV2B0",
            "checkpoint": str(
                PROJECT_ROOT
                / "models"
                / "advanced_v3_best.keras"
            ),
            "threshold": float(
                v3_metrics["threshold"]
            ),
            "threshold_source": (
                v3_metrics[
                    "threshold_source"
                ]
            ),
            "primary_candidate_status": "REJECTED",
            "automatic_override_status": "REJECTED",
            "automatic_ensemble_status": "REJECTED",
            "secondary_safety_signal_status": (
                "RETAINED_EXPLORATORY"
            ),
        },
        "held_out_v3_performance": {
            "samples": int(
                v3_metrics["samples"]
            ),
            "accuracy": float(
                v3_metrics["accuracy"]
            ),
            "precision": float(
                v3_metrics["precision"]
            ),
            "sensitivity": float(
                v3_metrics["sensitivity"]
            ),
            "specificity": float(
                v3_metrics["specificity"]
            ),
            "f1": float(
                v3_metrics["f1"]
            ),
            "roc_auc": float(
                v3_metrics["roc_auc"]
            ),
            "pr_auc": float(
                v3_metrics["pr_auc"]
            ),
            "false_positives": int(
                v3_metrics["false_positives"]
            ),
            "false_negatives": int(
                v3_metrics["false_negatives"]
            ),
        },
        "complementarity_findings": {
            "samples": int(
                complementarity["samples"]
            ),
            "disagreement_rate": float(
                complementarity[
                    "disagreement_rate"
                ]
            ),
            "probability_correlation": float(
                complementarity[
                    "probability_correlation"
                ]
            ),
            "v1_false_negatives": v1_fn,
            "v3_false_negatives": int(
                complementarity[
                    "v3_false_negatives"
                ]
            ),
            "shared_false_negatives": int(
                complementarity[
                    "shared_false_negatives"
                ]
            ),
            "v1_false_negatives_caught_by_v3": (
                v1_fn_caught_by_v3
            ),
            "v1_false_negative_rescue_rate": float(
                fn_rescue_rate
            ),
        },
        "deployment_policy": {
            "primary_prediction_source": (
                "baseline_cnn_v1"
            ),
            "secondary_signal_source": (
                "advanced_v3"
            ),
            "secondary_signal_is_exploratory": True,
            "automatic_override_allowed": False,
            "automatic_ensemble_allowed": False,
            "recommended_warning_condition": (
                "v1_predicts_normal_and_"
                "v3_predicts_pneumonia"
            ),
            "warning_action": (
                "flag_for_manual_review"
            ),
        },
        "methodological_note": (
            "The V3 secondary safety role was identified "
            "through held-out test error-overlap analysis. "
            "It is exploratory and is not prospectively or "
            "externally validated."
        ),
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            addendum,
            indent=2,
        )
    )

    print()
    print(
        f"Primary model:          "
        f"{selected_primary}"
    )
    print(
        "V3 primary candidate:   REJECTED"
    )
    print(
        "V3 safety signal:       "
        "RETAINED — EXPLORATORY"
    )
    print(
        "Automatic override:     REJECTED"
    )
    print(
        "Automatic ensemble:     REJECTED"
    )
    print(
        f"V1 FNs rescued by V3:   "
        f"{v1_fn_caught_by_v3}/{v1_fn} "
        f"({fn_rescue_rate:.1%})"
    )

    print(f"\nDecision addendum: {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print(
        "V3 MODEL SELECTION ADDENDUM STATUS: FROZEN"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
