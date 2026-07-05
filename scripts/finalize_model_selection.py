from __future__ import annotations

import json
from datetime import datetime

from backend.app.core.config import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_FILE = (
    RESULTS_DIR
    / "final_model_selection.json"
)


def main() -> None:

    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "FINAL MODEL SELECTION"
    )
    print("=" * 70)

    v1_metrics = json.loads(
        (
            RESULTS_DIR
            / "final_test_metrics.json"
        ).read_text()
    )

    v2_metrics = json.loads(
        (
            RESULTS_DIR
            / "final_test_v2_metrics.json"
        ).read_text()
    )

    ensemble_metrics = json.loads(
        (
            RESULTS_DIR
            / "final_ensemble_metrics.json"
        ).read_text()
    )

    calibrated_metrics = json.loads(
        (
            RESULTS_DIR
            / "final_calibrated_ensemble_metrics.json"
        ).read_text()
    )

    triage_metrics = json.loads(
        (
            RESULTS_DIR
            / "final_triage_metrics.json"
        ).read_text()
    )

    selection = {
        "decision_type": (
            "final_model_selection"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "selected_primary_model": {
            "name": "baseline_cnn_v1",

            "model_file": str(
                MODELS_DIR
                / "mediscan_final.keras"
            ),

            "decision_threshold": float(
                v1_metrics["threshold"]
            ),

            "selection_reason": (
                "Best held-out balance among "
                "evaluated production candidates."
            ),

            "test_metrics": {
                "accuracy": float(
                    v1_metrics["accuracy"]
                ),

                "precision": float(
                    v1_metrics["precision"]
                ),

                "sensitivity": float(
                    v1_metrics["sensitivity"]
                ),

                "specificity": float(
                    v1_metrics["specificity"]
                ),

                "f1": float(
                    v1_metrics["f1"]
                ),

                "roc_auc": float(
                    v1_metrics["roc_auc"]
                ),

                "pr_auc": float(
                    v1_metrics["pr_auc"]
                ),
            },
        },

        "auxiliary_components": {
            "v2_model": {
                "status": (
                    "analysis_only"
                ),

                "role": (
                    "secondary representation "
                    "and attention comparison"
                ),
            },

            "gradcam": {
                "status": (
                    "enabled"
                ),

                "role": (
                    "explanation and "
                    "shortcut-learning audit"
                ),

                "warning": (
                    "Not an anatomical "
                    "segmentation method."
                ),
            },

            "model_disagreement": {
                "status": (
                    "informational_only"
                ),

                "role": (
                    "uncertainty indicator"
                ),

                "warning": (
                    "Not validated as a "
                    "reliable safety gate."
                ),
            },
        },

        "rejected_production_candidates": {
            "v2_primary": {
                "reason": (
                    "Did not outperform V1 "
                    "on held-out classification."
                ),

                "accuracy": float(
                    v2_metrics["accuracy"]
                ),
            },

            "raw_ensemble": {
                "reason": (
                    "Reduced specificity and "
                    "overall accuracy."
                ),

                "accuracy": float(
                    ensemble_metrics["accuracy"]
                ),
            },

            "calibrated_ensemble": {
                "reason": (
                    "Calibration did not produce "
                    "superior held-out decision "
                    "performance."
                ),

                "accuracy": float(
                    calibrated_metrics[
                        "accuracy"
                    ]
                ),
            },

            "triage_safety_gate": {
                "reason": (
                    "Validation error-capture "
                    "performance did not "
                    "generalize to held-out test."
                ),

                "test_review_rate": float(
                    triage_metrics[
                        "review_rate"
                    ]
                ),

                "test_error_capture_rate": float(
                    triage_metrics[
                        "error_capture_rate"
                    ]
                ),
            },
        },

        "deployment_constraints": {
            "intended_use": (
                "educational decision-support "
                "prototype"
            ),

            "not_for_clinical_use": True,

            "human_review_required": True,

            "known_limitations": [
                (
                    "Single chest X-ray dataset."
                ),
                (
                    "Dataset shift observed "
                    "between validation and test."
                ),
                (
                    "False-positive rate remains "
                    "substantial."
                ),
                (
                    "Grad-CAM may attend to "
                    "non-anatomical regions."
                ),
            ],
        },
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            selection,
            indent=2,
        )
    )

    print(
        "\nSelected primary model: "
        "baseline_cnn_v1"
    )

    print(
        f"Threshold: "
        f"{v1_metrics['threshold']:.6f}"
    )

    print(
        "\nProduction candidates:"
    )

    print(
        "  V1 primary:          SELECTED"
    )

    print(
        "  V2 primary:          REJECTED"
    )

    print(
        "  Raw ensemble:        REJECTED"
    )

    print(
        "  Calibrated ensemble: REJECTED"
    )

    print(
        "  Triage safety gate:  REJECTED"
    )

    print(
        "\nAuxiliary components:"
    )

    print(
        "  V2:                  ANALYSIS ONLY"
    )

    print(
        "  Grad-CAM:            ENABLED"
    )

    print(
        "  Disagreement signal: INFORMATIONAL"
    )

    print(
        f"\nDecision record: {OUTPUT_FILE}"
    )

    print(
        "\nFINAL MODEL SELECTION "
        "STATUS: FROZEN"
    )


if __name__ == "__main__":
    main()