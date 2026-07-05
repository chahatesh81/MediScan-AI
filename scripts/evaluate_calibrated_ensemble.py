from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT

from scripts.calibrate_ensemble import (
    apply_platt,
    calculate_metrics,
    fit_platt_calibrator,
    validate_probabilities,
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

V1_VALIDATION_PATH = (
    RESULTS_DIR
    / "baseline_validation_predictions.csv"
)

V2_VALIDATION_PATH = (
    RESULTS_DIR
    / "baseline_v2_validation_predictions.csv"
)

V1_TEST_PATH = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)

V2_TEST_PATH = (
    RESULTS_DIR
    / "final_test_v2_predictions.csv"
)

OOF_ANALYSIS_PATH = (
    RESULTS_DIR
    / "ensemble_calibration_oof_analysis.json"
)

OUTPUT_METRICS_PATH = (
    RESULTS_DIR
    / "final_calibrated_ensemble_metrics.json"
)

OUTPUT_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "final_calibrated_ensemble_predictions.csv"
)


def main() -> None:

    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "FINAL PLATT-CALIBRATED ENSEMBLE BENCHMARK"
    )
    print("=" * 70)

    print(
        "\nLoading frozen OOF configuration..."
    )

    oof_analysis = json.loads(
        OOF_ANALYSIS_PATH.read_text()
    )

    selected_method = (
        oof_analysis["selected_method"]
    )

    if selected_method != "platt_oof":
        raise RuntimeError(
            "Expected OOF-selected method "
            f"'platt_oof', got "
            f"'{selected_method}'."
        )

    configuration = (
        oof_analysis[
            "selected_configuration"
        ]
    )

    v1_weight = float(
        configuration["v1_weight"]
    )

    v2_weight = float(
        configuration["v2_weight"]
    )

    threshold = float(
        configuration["threshold"]
    )

    print(
        f"Selected method:  {selected_method}"
    )

    print(
        f"V1 weight:        {v1_weight:.6f}"
    )

    print(
        f"V2 weight:        {v2_weight:.6f}"
    )

    print(
        f"Frozen threshold: {threshold:.6f}"
    )

    if not np.isclose(
        v1_weight + v2_weight,
        1.0,
    ):
        raise RuntimeError(
            "Ensemble weights do not sum to 1."
        )

    print(
        "\nLoading validation predictions..."
    )

    v1_val = pd.read_csv(
        V1_VALIDATION_PATH
    )

    v2_val = pd.read_csv(
        V2_VALIDATION_PATH
    )

    if len(v1_val) != len(v2_val):
        raise RuntimeError(
            "Validation sample counts "
            "do not match."
        )

    y_val_v1 = (
        v1_val["true_label"]
        .to_numpy(dtype=np.int32)
    )

    y_val_v2 = (
        v2_val["true_label"]
        .to_numpy(dtype=np.int32)
    )

    if not np.array_equal(
        y_val_v1,
        y_val_v2,
    ):
        raise RuntimeError(
            "Validation label order "
            "does not match."
        )

    v1_val_probability = (
        v1_val["probability"]
        .to_numpy(dtype=np.float64)
    )

    v2_val_probability = (
        v2_val["probability"]
        .to_numpy(dtype=np.float64)
    )

    validate_probabilities(
        v1_val_probability,
        "V1 validation probabilities",
    )

    validate_probabilities(
        v2_val_probability,
        "V2 validation probabilities",
    )

    print(
        f"Validation samples: {len(y_val_v1):,}"
    )

    print(
        "Validation alignment: PASS"
    )

    print(
        "\nFitting final Platt calibrators "
        "on full validation set..."
    )

    v1_platt_model = (
        fit_platt_calibrator(
            v1_val_probability,
            y_val_v1,
        )
    )

    v2_platt_model = (
        fit_platt_calibrator(
            v2_val_probability,
            y_val_v1,
        )
    )

    print(
        "Final Platt calibrators: READY"
    )

    print(
        "\nLoading saved test predictions..."
    )

    v1_test = pd.read_csv(
        V1_TEST_PATH
    )

    v2_test = pd.read_csv(
        V2_TEST_PATH
    )

    print(
        f"V1 test rows: {len(v1_test):,}"
    )

    print(
        f"V2 test rows: {len(v2_test):,}"
    )

    if len(v1_test) != len(v2_test):
        raise RuntimeError(
            "V1/V2 test sample counts "
            "do not match."
        )

    y_test_v1 = (
        v1_test["true_label"]
        .to_numpy(dtype=np.int32)
    )

    y_test_v2 = (
        v2_test["true_label"]
        .to_numpy(dtype=np.int32)
    )

    labels_match = np.array_equal(
        y_test_v1,
        y_test_v2,
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

    y_true = y_test_v1

    v1_test_raw = (
        v1_test["probability"]
        .to_numpy(dtype=np.float64)
    )

    v2_test_raw = (
        v2_test["probability"]
        .to_numpy(dtype=np.float64)
    )

    validate_probabilities(
        v1_test_raw,
        "V1 test probabilities",
    )

    validate_probabilities(
        v2_test_raw,
        "V2 test probabilities",
    )

    print(
        "Test probability checks: PASS"
    )

    print(
        "\nApplying frozen calibration pipeline..."
    )

    v1_test_calibrated = apply_platt(
        v1_platt_model,
        v1_test_raw,
    )

    v2_test_calibrated = apply_platt(
        v2_platt_model,
        v2_test_raw,
    )

    ensemble_probability = (
        v1_weight
        * v1_test_calibrated
        +
        v2_weight
        * v2_test_calibrated
    )

    validate_probabilities(
        ensemble_probability,
        "Calibrated ensemble probabilities",
    )

    metrics = calculate_metrics(
        y_true,
        ensemble_probability,
        threshold,
    )

    output = {
        "evaluation_type": (
            "final_platt_calibrated_"
            "ensemble_benchmark"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "selection_source": (
            "five_fold_oof_validation"
        ),

        "calibration_method": (
            "platt_scaling"
        ),

        "calibration_fit_source": (
            "full_validation_set"
        ),

        "v1_weight": v1_weight,

        "v2_weight": v2_weight,

        "threshold": threshold,

        "threshold_source": (
            "oof_validation_youden"
        ),

        "samples": int(
            len(y_true)
        ),

        "v1_platt_parameters": {
            "coefficient": float(
                v1_platt_model.coef_[0, 0]
            ),

            "intercept": float(
                v1_platt_model.intercept_[0]
            ),
        },

        "v2_platt_parameters": {
            "coefficient": float(
                v2_platt_model.coef_[0, 0]
            ),

            "intercept": float(
                v2_platt_model.intercept_[0]
            ),
        },

        **metrics,
    }

    OUTPUT_METRICS_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    predictions = (
        ensemble_probability
        >= threshold
    ).astype(np.int32)

    pd.DataFrame(
        {
            "true_label": y_true,

            "v1_raw_probability": (
                v1_test_raw
            ),

            "v2_raw_probability": (
                v2_test_raw
            ),

            "v1_platt_probability": (
                v1_test_calibrated
            ),

            "v2_platt_probability": (
                v2_test_calibrated
            ),

            "ensemble_probability": (
                ensemble_probability
            ),

            "predicted_label": (
                predictions
            ),
        }
    ).to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL CALIBRATED ENSEMBLE RESULTS"
    )

    print(
        "=" * 70
    )

    for name, value in output.items():

        if isinstance(
            value,
            float,
        ):
            print(
                f"{name:24s}: "
                f"{value:.4f}"
            )

        elif not isinstance(
            value,
            dict,
        ):
            print(
                f"{name:24s}: "
                f"{value}"
            )

    print(
        "\nPlatt parameters:"
    )

    print(
        "V1 coefficient: "
        f"{output['v1_platt_parameters']['coefficient']:.6f}"
    )

    print(
        "V1 intercept:   "
        f"{output['v1_platt_parameters']['intercept']:.6f}"
    )

    print(
        "V2 coefficient: "
        f"{output['v2_platt_parameters']['coefficient']:.6f}"
    )

    print(
        "V2 intercept:   "
        f"{output['v2_platt_parameters']['intercept']:.6f}"
    )

    print(
        f"\nMetrics:     "
        f"{OUTPUT_METRICS_PATH}"
    )

    print(
        f"Predictions: "
        f"{OUTPUT_PREDICTIONS_PATH}"
    )

    print(
        "\nFINAL PLATT-CALIBRATED "
        "ENSEMBLE BENCHMARK STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()