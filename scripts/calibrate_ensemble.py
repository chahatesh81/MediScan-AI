from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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

OUTPUT_JSON = (
    RESULTS_DIR
    / "ensemble_calibration_analysis.json"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "ensemble_calibration_search.csv"
)

CALIBRATED_PREDICTIONS_CSV = (
    RESULTS_DIR
    / "ensemble_calibrated_validation_predictions.csv"
)

RANDOM_SEED = 42

WEIGHTS = np.linspace(
    0.0,
    1.0,
    101,
)


def validate_probabilities(
    probabilities: np.ndarray,
    name: str,
) -> None:

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            f"{name} contains non-finite values."
        )

    if np.any(
        probabilities < 0.0
    ) or np.any(
        probabilities > 1.0
    ):
        raise RuntimeError(
            f"{name} contains values "
            f"outside [0, 1]."
        )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

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
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "sensitivity": float(
            recall_score(
                y_true,
                predictions,
            )
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
            )
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

        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                np.clip(
                    probabilities,
                    1e-7,
                    1.0 - 1e-7,
                ),
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

    false_positive_rate, (
        true_positive_rate
    ), thresholds = roc_curve(
        y_true,
        probabilities,
    )

    valid = np.isfinite(
        thresholds
    )

    false_positive_rate = (
        false_positive_rate[valid]
    )

    true_positive_rate = (
        true_positive_rate[valid]
    )

    thresholds = thresholds[valid]

    youden_j = (
        true_positive_rate
        - false_positive_rate
    )

    best_index = int(
        np.argmax(youden_j)
    )

    return float(
        thresholds[best_index]
    )


def fit_platt_calibrator(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:

    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
    )

    model.fit(
        probabilities.reshape(-1, 1),
        y_true,
    )

    return model


def apply_platt(
    model: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:

    calibrated = model.predict_proba(
        probabilities.reshape(-1, 1)
    )[:, 1]

    return calibrated.astype(
        np.float64
    )


def fit_isotonic_calibrator(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> IsotonicRegression:

    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    model.fit(
        probabilities,
        y_true,
    )

    return model


def search_ensemble(
    y_true: np.ndarray,
    v1_probabilities: np.ndarray,
    v2_probabilities: np.ndarray,
    calibration_method: str,
) -> tuple[
    dict,
    list[dict],
]:

    records = []

    best_result = None
    best_score = None

    for v1_weight in WEIGHTS:

        v2_weight = (
            1.0 - v1_weight
        )

        ensemble_probability = (
            v1_weight
            * v1_probabilities
            +
            v2_weight
            * v2_probabilities
        )

        threshold = (
            find_youden_threshold(
                y_true,
                ensemble_probability,
            )
        )

        metrics = calculate_metrics(
            y_true,
            ensemble_probability,
            threshold,
        )

        record = {
            "calibration_method": (
                calibration_method
            ),
            "v1_weight": float(
                v1_weight
            ),
            "v2_weight": float(
                v2_weight
            ),
            **metrics,
        }

        records.append(
            record
        )

        score = (
            metrics["sensitivity"]
            + metrics["specificity"],
            metrics["f1"],
            metrics["roc_auc"],
            -metrics["brier_score"],
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score
            best_result = record

    if best_result is None:
        raise RuntimeError(
            "Ensemble search failed."
        )

    return (
        best_result,
        records,
    )


def print_result(
    name: str,
    result: dict,
) -> None:

    print(f"\n{name}")
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
        f"  Brier score:  "
        f"{result['brier_score']:.6f}"
    )
    print(
        f"  Log loss:     "
        f"{result['log_loss']:.6f}"
    )
    print(
        "  TN/FP/FN/TP:  "
        f"{result['tn']}/"
        f"{result['fp']}/"
        f"{result['fn']}/"
        f"{result['tp']}"
    )


def main() -> None:

    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "ENSEMBLE CALIBRATION ANALYSIS"
    )
    print("=" * 70)

    print("\nLoading validation predictions...")

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
            "V1/V2 validation sample "
            "counts do not match."
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

    y_true = y_true_v1

    v1_raw = (
        v1["probability"]
        .to_numpy(dtype=np.float64)
    )

    v2_raw = (
        v2["probability"]
        .to_numpy(dtype=np.float64)
    )

    validate_probabilities(
        v1_raw,
        "V1 probabilities",
    )

    validate_probabilities(
        v2_raw,
        "V2 probabilities",
    )

    print(
        "Probability checks: PASS"
    )

    print(
        "\nFitting Platt calibrators..."
    )

    v1_platt_model = (
        fit_platt_calibrator(
            v1_raw,
            y_true,
        )
    )

    v2_platt_model = (
        fit_platt_calibrator(
            v2_raw,
            y_true,
        )
    )

    v1_platt = apply_platt(
        v1_platt_model,
        v1_raw,
    )

    v2_platt = apply_platt(
        v2_platt_model,
        v2_raw,
    )

    print(
        "Fitting isotonic calibrators..."
    )

    v1_isotonic_model = (
        fit_isotonic_calibrator(
            v1_raw,
            y_true,
        )
    )

    v2_isotonic_model = (
        fit_isotonic_calibrator(
            v2_raw,
            y_true,
        )
    )

    v1_isotonic = (
        v1_isotonic_model.predict(
            v1_raw
        )
    )

    v2_isotonic = (
        v2_isotonic_model.predict(
            v2_raw
        )
    )

    print(
        "\nSearching raw ensemble..."
    )

    raw_best, raw_records = (
        search_ensemble(
            y_true,
            v1_raw,
            v2_raw,
            "raw",
        )
    )

    print(
        "Searching Platt-calibrated "
        "ensemble..."
    )

    platt_best, platt_records = (
        search_ensemble(
            y_true,
            v1_platt,
            v2_platt,
            "platt",
        )
    )

    print(
        "Searching isotonic-calibrated "
        "ensemble..."
    )

    (
        isotonic_best,
        isotonic_records,
    ) = search_ensemble(
        y_true,
        v1_isotonic,
        v2_isotonic,
        "isotonic",
    )

    all_best = {
        "raw": raw_best,
        "platt": platt_best,
        "isotonic": isotonic_best,
    }

    selected_method = max(
        all_best,
        key=lambda method: (
            all_best[method][
                "sensitivity"
            ]
            +
            all_best[method][
                "specificity"
            ],
            all_best[method]["f1"],
            -all_best[method][
                "brier_score"
            ],
        ),
    )

    selected = all_best[
        selected_method
    ]

    print("\n" + "=" * 70)
    print(
        "CALIBRATION RESULTS"
    )
    print("=" * 70)

    print_result(
        "RAW",
        raw_best,
    )

    print_result(
        "PLATT",
        platt_best,
    )

    print_result(
        "ISOTONIC",
        isotonic_best,
    )

    print("\n" + "=" * 70)
    print(
        "SELECTED VALIDATION CONFIGURATION"
    )
    print("=" * 70)

    print(
        f"Calibration method: "
        f"{selected_method}"
    )

    print_result(
        "SELECTED",
        selected,
    )

    all_records = (
        raw_records
        + platt_records
        + isotonic_records
    )

    pd.DataFrame(
        all_records
    ).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    pd.DataFrame(
        {
            "true_label": y_true,

            "v1_raw_probability": (
                v1_raw
            ),

            "v2_raw_probability": (
                v2_raw
            ),

            "v1_platt_probability": (
                v1_platt
            ),

            "v2_platt_probability": (
                v2_platt
            ),

            "v1_isotonic_probability": (
                v1_isotonic
            ),

            "v2_isotonic_probability": (
                v2_isotonic
            ),
        }
    ).to_csv(
        CALIBRATED_PREDICTIONS_CSV,
        index=False,
    )

    output = {
        "evaluation_type": (
            "validation_ensemble_calibration"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "selection_rule": (
            "validation_only_calibration_"
            "weight_threshold_search"
        ),

        "samples": int(
            len(y_true)
        ),

        "weights_tested": int(
            len(WEIGHTS)
        ),

        "methods": all_best,

        "selected_method": (
            selected_method
        ),

        "selected_configuration": (
            selected
        ),

        "platt_parameters": {
            "v1": {
                "coefficient": float(
                    v1_platt_model.coef_[
                        0, 0
                    ]
                ),
                "intercept": float(
                    v1_platt_model.intercept_[
                        0
                    ]
                ),
            },

            "v2": {
                "coefficient": float(
                    v2_platt_model.coef_[
                        0, 0
                    ]
                ),
                "intercept": float(
                    v2_platt_model.intercept_[
                        0
                    ]
                ),
            },
        },
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
        f"CSV:  {OUTPUT_CSV}"
    )

    print(
        "Calibrated predictions: "
        f"{CALIBRATED_PREDICTIONS_CSV}"
    )

    print(
        "\nENSEMBLE CALIBRATION "
        "ANALYSIS STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()