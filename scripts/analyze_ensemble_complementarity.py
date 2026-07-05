import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT


V1_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_validation_predictions.csv"
)

V2_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "baseline_v2_validation_predictions.csv"
)


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "V1/V2 ENSEMBLE COMPLEMENTARITY AUDIT"
    )
    print("=" * 70)

    v1 = pd.read_csv(V1_FILE)
    v2 = pd.read_csv(V2_FILE)

    if len(v1) != len(v2):
        raise RuntimeError(
            "Prediction row counts do not match."
        )

    if not np.array_equal(
        v1["true_label"].to_numpy(),
        v2["true_label"].to_numpy(),
    ):
        raise RuntimeError(
            "V1 and V2 validation rows "
            "are not aligned."
        )

    y_true = v1[
        "true_label"
    ].to_numpy(dtype=int)

    p1 = v1[
        "probability"
    ].to_numpy(dtype=float)

    p2 = v2[
        "probability"
    ].to_numpy(dtype=float)

    v1_pred = v1[
        "predicted_label"
    ].to_numpy(dtype=int)

    v2_pred = v2[
        "predicted_label"
    ].to_numpy(dtype=int)

    v1_correct = (
        v1_pred == y_true
    )

    v2_correct = (
        v2_pred == y_true
    )

    both_correct = int(
        np.sum(
            v1_correct & v2_correct
        )
    )

    only_v1_correct = int(
        np.sum(
            v1_correct & ~v2_correct
        )
    )

    only_v2_correct = int(
        np.sum(
            ~v1_correct & v2_correct
        )
    )

    both_wrong = int(
        np.sum(
            ~v1_correct & ~v2_correct
        )
    )

    probability_correlation = float(
        np.corrcoef(
            p1,
            p2,
        )[0, 1]
    )

    print()
    print(
        f"Samples:                  "
        f"{len(y_true)}"
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
        f"Only V2 correct:          "
        f"{only_v2_correct}"
    )

    print(
        f"Both wrong:               "
        f"{both_wrong}"
    )

    print(
        f"Probability correlation:  "
        f"{probability_correlation:.4f}"
    )

    print()
    print(
        "Potential oracle-correct cases: "
        f"{both_correct + only_v1_correct + only_v2_correct}"
    )

    print()
    print(
        "ENSEMBLE COMPLEMENTARITY "
        "AUDIT STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()