from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from backend.app.core.config import PROJECT_ROOT

from scripts.calibrate_ensemble import (
    apply_platt,
    calculate_metrics,
    find_youden_threshold,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    search_ensemble,
    validate_probabilities,
)


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
    / "ensemble_calibration_oof_analysis.json"
)

OUTPUT_SEARCH_CSV = (
    RESULTS_DIR
    / "ensemble_calibration_oof_search.csv"
)

OUTPUT_PREDICTIONS_CSV = (
    RESULTS_DIR
    / "ensemble_calibration_oof_predictions.csv"
)

N_SPLITS = 5
RANDOM_SEED = 42


def build_oof_calibrated_probabilities(
    y_true: np.ndarray,
    v1_raw: np.ndarray,
    v2_raw: np.ndarray,
) -> dict[str, np.ndarray]:

    sample_count = len(y_true)

    v1_platt_oof = np.full(
        sample_count,
        np.nan,
        dtype=np.float64,
    )

    v2_platt_oof = np.full(
        sample_count,
        np.nan,
        dtype=np.float64,
    )

    v1_isotonic_oof = np.full(
        sample_count,
        np.nan,
        dtype=np.float64,
    )

    v2_isotonic_oof = np.full(
        sample_count,
        np.nan,
        dtype=np.float64,
    )

    fold_ids = np.full(
        sample_count,
        -1,
        dtype=np.int32,
    )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    print(
        f"\nBuilding {N_SPLITS}-fold "
        "out-of-fold calibrated probabilities..."
    )

    for fold_number, (
        train_indices,
        holdout_indices,
    ) in enumerate(
        splitter.split(
            v1_raw,
            y_true,
        ),
        start=1,
    ):

        print(
            f"\nFold {fold_number}/{N_SPLITS}"
        )

        print(
            f"  Calibration fit samples: "
            f"{len(train_indices)}"
        )

        print(
            f"  OOF holdout samples:      "
            f"{len(holdout_indices)}"
        )

        y_train = y_true[
            train_indices
        ]

        v1_train = v1_raw[
            train_indices
        ]

        v2_train = v2_raw[
            train_indices
        ]

        v1_holdout = v1_raw[
            holdout_indices
        ]

        v2_holdout = v2_raw[
            holdout_indices
        ]

        v1_platt_model = (
            fit_platt_calibrator(
                v1_train,
                y_train,
            )
        )

        v2_platt_model = (
            fit_platt_calibrator(
                v2_train,
                y_train,
            )
        )

        v1_platt_oof[
            holdout_indices
        ] = apply_platt(
            v1_platt_model,
            v1_holdout,
        )

        v2_platt_oof[
            holdout_indices
        ] = apply_platt(
            v2_platt_model,
            v2_holdout,
        )

        v1_isotonic_model = (
            fit_isotonic_calibrator(
                v1_train,
                y_train,
            )
        )

        v2_isotonic_model = (
            fit_isotonic_calibrator(
                v2_train,
                y_train,
            )
        )

        v1_isotonic_oof[
            holdout_indices
        ] = (
            v1_isotonic_model.predict(
                v1_holdout
            )
        )

        v2_isotonic_oof[
            holdout_indices
        ] = (
            v2_isotonic_model.predict(
                v2_holdout
            )
        )

        fold_ids[
            holdout_indices
        ] = fold_number

    outputs = {
        "fold_id": fold_ids,
        "v1_platt": v1_platt_oof,
        "v2_platt": v2_platt_oof,
        "v1_isotonic": v1_isotonic_oof,
        "v2_isotonic": v2_isotonic_oof,
    }

    for name, values in outputs.items():

        if name == "fold_id":
            continue

        if not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(
                f"{name} OOF predictions "
                "contain missing or non-finite values."
            )

        validate_probabilities(
            values,
            name,
        )

    if np.any(
        fold_ids < 1
    ):
        raise RuntimeError(
            "One or more samples were not "
            "assigned to an OOF fold."
        )

    print(
        "\nOOF coverage check: PASS"
    )

    return outputs


def print_method_result(
    method_name: str,
    result: dict,
) -> None:

    print(
        f"\n{method_name.upper()}"
    )

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
        "OOF ENSEMBLE CALIBRATION VALIDATION"
    )
    print("=" * 70)

    print(
        "\nLoading validation predictions..."
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
        "V1 raw probabilities",
    )

    validate_probabilities(
        v2_raw,
        "V2 raw probabilities",
    )

    print(
        "Probability checks: PASS"
    )

    oof = (
        build_oof_calibrated_probabilities(
            y_true=y_true,
            v1_raw=v1_raw,
            v2_raw=v2_raw,
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
        "Searching OOF Platt ensemble..."
    )

    platt_best, platt_records = (
        search_ensemble(
            y_true,
            oof["v1_platt"],
            oof["v2_platt"],
            "platt_oof",
        )
    )

    print(
        "Searching OOF isotonic ensemble..."
    )

    (
        isotonic_best,
        isotonic_records,
    ) = search_ensemble(
        y_true,
        oof["v1_isotonic"],
        oof["v2_isotonic"],
        "isotonic_oof",
    )

    methods = {
        "raw": raw_best,
        "platt_oof": platt_best,
        "isotonic_oof": isotonic_best,
    }

    print(
        "\n" + "=" * 70
    )

    print(
        "OOF CALIBRATION RESULTS"
    )

    print(
        "=" * 70
    )

    for (
        method_name,
        result,
    ) in methods.items():

        print_method_result(
            method_name,
            result,
        )

    selected_method = max(
        methods,
        key=lambda method: (
            methods[method][
                "sensitivity"
            ]
            +
            methods[method][
                "specificity"
            ],
            methods[method]["f1"],
            methods[method]["roc_auc"],
            -methods[method][
                "brier_score"
            ],
        ),
    )

    selected = methods[
        selected_method
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED OOF CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Method: {selected_method}"
    )

    print_method_result(
        "selected",
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
        OUTPUT_SEARCH_CSV,
        index=False,
    )

    pd.DataFrame(
        {
            "true_label": y_true,
            "fold_id": oof["fold_id"],
            "v1_raw_probability": v1_raw,
            "v2_raw_probability": v2_raw,
            "v1_platt_oof_probability": (
                oof["v1_platt"]
            ),
            "v2_platt_oof_probability": (
                oof["v2_platt"]
            ),
            "v1_isotonic_oof_probability": (
                oof["v1_isotonic"]
            ),
            "v2_isotonic_oof_probability": (
                oof["v2_isotonic"]
            ),
        }
    ).to_csv(
        OUTPUT_PREDICTIONS_CSV,
        index=False,
    )

    output = {
        "evaluation_type": (
            "validation_oof_ensemble_calibration"
        ),
        "timestamp": (
            datetime.now().isoformat()
        ),
        "selection_rule": (
            "five_fold_stratified_oof_"
            "calibration_then_validation_"
            "weight_threshold_search"
        ),
        "samples": int(
            len(y_true)
        ),
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
        "methods": methods,
        "selected_method": (
            selected_method
        ),
        "selected_configuration": (
            selected
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
        f"Search CSV: {OUTPUT_SEARCH_CSV}"
    )

    print(
        "OOF predictions: "
        f"{OUTPUT_PREDICTIONS_CSV}"
    )

    print(
        "\nOOF ENSEMBLE CALIBRATION "
        "VALIDATION STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()