from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

V1_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)

V3_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "final_test_v3_predictions.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "v1_v3_complementarity_audit.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "v1_v3_complementarity_audit.json"
)


def validate_predictions(
    dataframe: pd.DataFrame,
    model_name: str,
) -> None:
    required_columns = {
        "true_label",
        "probability",
        "predicted_label",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"{model_name} missing columns: "
            f"{sorted(missing_columns)}"
        )

    probabilities = dataframe[
        "probability"
    ].to_numpy(dtype=np.float64)

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            f"{model_name} contains "
            "non-finite probabilities."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise RuntimeError(
            f"{model_name} probabilities "
            "outside [0, 1]."
        )


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "V1/V3 COMPLEMENTARITY AND ERROR-OVERLAP AUDIT"
    )
    print("=" * 70)

    print("\nLoading held-out test predictions...")

    v1 = pd.read_csv(
        V1_PREDICTIONS_FILE
    )

    v3 = pd.read_csv(
        V3_PREDICTIONS_FILE
    )

    print(f"V1 rows: {len(v1)}")
    print(f"V3 rows: {len(v3)}")

    if len(v1) != len(v3):
        raise RuntimeError(
            "V1/V3 sample counts do not match."
        )

    validate_predictions(
        v1,
        "V1",
    )

    validate_predictions(
        v3,
        "V3",
    )

    labels_match = np.array_equal(
        v1["true_label"].to_numpy(),
        v3["true_label"].to_numpy(),
    )

    print(
        "Label order: "
        f"{'PASS' if labels_match else 'FAIL'}"
    )

    if not labels_match:
        raise RuntimeError(
            "V1/V3 true-label order mismatch."
        )

    audit = pd.DataFrame(
        {
            "true_label": (
                v1["true_label"]
                .to_numpy(dtype=np.int32)
            ),
            "v1_probability": (
                v1["probability"]
                .to_numpy(dtype=np.float64)
            ),
            "v1_prediction": (
                v1["predicted_label"]
                .to_numpy(dtype=np.int32)
            ),
            "v3_probability": (
                v3["probability"]
                .to_numpy(dtype=np.float64)
            ),
            "v3_prediction": (
                v3["predicted_label"]
                .to_numpy(dtype=np.int32)
            ),
        }
    )

    audit["v1_correct"] = (
        audit["v1_prediction"]
        == audit["true_label"]
    )

    audit["v3_correct"] = (
        audit["v3_prediction"]
        == audit["true_label"]
    )

    audit["models_disagree"] = (
        audit["v1_prediction"]
        != audit["v3_prediction"]
    )

    audit["v1_false_negative"] = (
        (audit["true_label"] == 1)
        & (audit["v1_prediction"] == 0)
    )

    audit["v3_false_negative"] = (
        (audit["true_label"] == 1)
        & (audit["v3_prediction"] == 0)
    )

    audit["v1_false_positive"] = (
        (audit["true_label"] == 0)
        & (audit["v1_prediction"] == 1)
    )

    audit["v3_false_positive"] = (
        (audit["true_label"] == 0)
        & (audit["v3_prediction"] == 1)
    )

    audit["error_overlap_type"] = np.select(
        [
            (
                audit["v1_correct"]
                & audit["v3_correct"]
            ),
            (
                ~audit["v1_correct"]
                & audit["v3_correct"]
            ),
            (
                audit["v1_correct"]
                & ~audit["v3_correct"]
            ),
        ],
        [
            "both_correct",
            "only_v3_correct",
            "only_v1_correct",
        ],
        default="both_wrong",
    )

    both_correct = int(
        (
            audit["error_overlap_type"]
            == "both_correct"
        ).sum()
    )

    only_v1_correct = int(
        (
            audit["error_overlap_type"]
            == "only_v1_correct"
        ).sum()
    )

    only_v3_correct = int(
        (
            audit["error_overlap_type"]
            == "only_v3_correct"
        ).sum()
    )

    both_wrong = int(
        (
            audit["error_overlap_type"]
            == "both_wrong"
        ).sum()
    )

    disagreements = int(
        audit["models_disagree"].sum()
    )

    v1_fn_count = int(
        audit["v1_false_negative"].sum()
    )

    v3_fn_count = int(
        audit["v3_false_negative"].sum()
    )

    shared_fn_count = int(
        (
            audit["v1_false_negative"]
            & audit["v3_false_negative"]
        ).sum()
    )

    v1_fn_caught_by_v3 = int(
        (
            audit["v1_false_negative"]
            & ~audit["v3_false_negative"]
        ).sum()
    )

    v3_fn_caught_by_v1 = int(
        (
            audit["v3_false_negative"]
            & ~audit["v1_false_negative"]
        ).sum()
    )

    v1_fp_count = int(
        audit["v1_false_positive"].sum()
    )

    v3_fp_count = int(
        audit["v3_false_positive"].sum()
    )

    shared_fp_count = int(
        (
            audit["v1_false_positive"]
            & audit["v3_false_positive"]
        ).sum()
    )

    probability_correlation = float(
        np.corrcoef(
            audit["v1_probability"],
            audit["v3_probability"],
        )[0, 1]
    )

    summary = {
        "evaluation_type": (
            "v1_v3_held_out_complementarity_audit"
        ),
        "samples": int(len(audit)),
        "both_correct": both_correct,
        "only_v1_correct": only_v1_correct,
        "only_v3_correct": only_v3_correct,
        "both_wrong": both_wrong,
        "model_disagreements": disagreements,
        "disagreement_rate": float(
            disagreements / len(audit)
        ),
        "probability_correlation": (
            probability_correlation
        ),
        "v1_false_negatives": v1_fn_count,
        "v3_false_negatives": v3_fn_count,
        "shared_false_negatives": (
            shared_fn_count
        ),
        "v1_false_negatives_caught_by_v3": (
            v1_fn_caught_by_v3
        ),
        "v3_false_negatives_caught_by_v1": (
            v3_fn_caught_by_v1
        ),
        "v1_false_positives": v1_fp_count,
        "v3_false_positives": v3_fp_count,
        "shared_false_positives": (
            shared_fp_count
        ),
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

    print("\n" + "=" * 70)
    print("ERROR OVERLAP")
    print("=" * 70)

    print(
        f"Samples:                  "
        f"{len(audit)}"
    )
    print(
        f"Both correct:             "
        f"{both_correct}"
    )
    print(
        f"Only V1 correct:          "
        f"{only_v1_correct}"
    )
    print(
        f"Only V3 correct:          "
        f"{only_v3_correct}"
    )
    print(
        f"Both wrong:               "
        f"{both_wrong}"
    )
    print(
        f"Model disagreements:      "
        f"{disagreements}"
    )
    print(
        f"Disagreement rate:        "
        f"{disagreements / len(audit):.4f}"
    )
    print(
        f"Probability correlation:  "
        f"{probability_correlation:.4f}"
    )

    print("\n" + "=" * 70)
    print("FALSE-NEGATIVE OVERLAP")
    print("=" * 70)

    print(
        f"V1 false negatives:       "
        f"{v1_fn_count}"
    )
    print(
        f"V3 false negatives:       "
        f"{v3_fn_count}"
    )
    print(
        f"Shared false negatives:   "
        f"{shared_fn_count}"
    )
    print(
        f"V1 FNs caught by V3:      "
        f"{v1_fn_caught_by_v3}"
    )
    print(
        f"V3 FNs caught by V1:      "
        f"{v3_fn_caught_by_v1}"
    )

    print("\n" + "=" * 70)
    print("FALSE-POSITIVE OVERLAP")
    print("=" * 70)

    print(
        f"V1 false positives:       "
        f"{v1_fp_count}"
    )
    print(
        f"V3 false positives:       "
        f"{v3_fp_count}"
    )
    print(
        f"Shared false positives:   "
        f"{shared_fp_count}"
    )

    print(f"\nCSV:  {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")

    print(
        "\nV1/V3 COMPLEMENTARITY AUDIT "
        "STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()