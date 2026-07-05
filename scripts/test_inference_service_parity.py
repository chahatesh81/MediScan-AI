from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT
from backend.app.services.inference_service import (
    inference_service,
)


FINAL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_manifest.csv"
)

V1_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_test_predictions.csv"
)

V3_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_test_v3_predictions.csv"
)

RAW_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
)

ROW_INDEX = 0

V1_ATOL = 1e-5
V3_ATOL = 2e-4


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "PRODUCTION INFERENCE PARITY TEST"
    )
    print("=" * 70)

    manifest = pd.read_csv(
        FINAL_MANIFEST
    )

    test_manifest = (
        manifest[
            manifest["final_split"]
            == "test"
        ]
        .reset_index(drop=True)
        .copy()
    )

    v1_predictions = pd.read_csv(
        V1_PREDICTIONS
    )

    v3_predictions = pd.read_csv(
        V3_PREDICTIONS
    )

    if not (
        len(test_manifest)
        == len(v1_predictions)
        == len(v3_predictions)
    ):
        raise RuntimeError(
            "Test sample counts are not aligned."
        )

    expected_labels = (
        test_manifest["class_name"]
        .map(
            {
                "NORMAL": 0,
                "PNEUMONIA": 1,
            }
        )
        .to_numpy(dtype=np.int32)
    )

    v1_labels = (
        v1_predictions["true_label"]
        .to_numpy(dtype=np.int32)
    )

    v3_labels = (
        v3_predictions["true_label"]
        .to_numpy(dtype=np.int32)
    )

    if not np.array_equal(
        expected_labels,
        v1_labels,
    ):
        raise RuntimeError(
            "V1 prediction order is not aligned "
            "with the test manifest."
        )

    if not np.array_equal(
        expected_labels,
        v3_labels,
    ):
        raise RuntimeError(
            "V3 prediction order is not aligned "
            "with the test manifest."
        )

    row = test_manifest.iloc[
        ROW_INDEX
    ]

    image_path = (
        RAW_IMAGE_ROOT
        / row["path"]
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            image_path
        )

    print(f"Row index:      {ROW_INDEX}")
    print(f"Image:          {image_path}")
    print(
        f"True label:     "
        f"{row['class_name']}"
    )

    result = (
        inference_service.predict_bytes(
            image_path.read_bytes()
        )
    )

    actual_v1 = float(
        result[
            "primary_prediction"
        ][
            "probability"
        ]
    )

    actual_v3 = float(
        result[
            "secondary_signal"
        ][
            "probability"
        ]
    )

    expected_v1 = float(
        v1_predictions.iloc[
            ROW_INDEX
        ][
            "probability"
        ]
    )

    expected_v3 = float(
        v3_predictions.iloc[
            ROW_INDEX
        ][
            "probability"
        ]
    )

    v1_difference = abs(
        actual_v1
        - expected_v1
    )

    v3_difference = abs(
        actual_v3
        - expected_v3
    )

    v1_pass = bool(
        np.isclose(
            actual_v1,
            expected_v1,
            atol=V1_ATOL,
            rtol=0.0,
        )
    )

    v3_pass = bool(
        np.isclose(
            actual_v3,
            expected_v3,
            atol=V3_ATOL,
            rtol=0.0,
        )
    )

    print()
    print("V1 PARITY")
    print(
        f"  Expected:    "
        f"{expected_v1:.10f}"
    )
    print(
        f"  Production:  "
        f"{actual_v1:.10f}"
    )
    print(
        f"  Difference:  "
        f"{v1_difference:.10e}"
    )
    print(
        f"  Status:      "
        f"{'PASS' if v1_pass else 'FAIL'}"
    )

    print()
    print("V3 PARITY")
    print(
        f"  Expected:    "
        f"{expected_v3:.10f}"
    )
    print(
        f"  Production:  "
        f"{actual_v3:.10f}"
    )
    print(
        f"  Difference:  "
        f"{v3_difference:.10e}"
    )
    print(
        f"  Status:      "
        f"{'PASS' if v3_pass else 'FAIL'}"
    )

    print()
    print("PRODUCTION DECISION")
    print(
        "  Final label: "
        f"{result['decision']['final_label']}"
    )
    print(
        "  Manual review: "
        f"{result['decision']['manual_review_recommended']}"
    )
    print(
        "  Warning code: "
        f"{result['decision']['warning_code']}"
    )

    if not v1_pass:
        raise RuntimeError(
            "V1 production inference does not "
            "match frozen evaluation inference."
        )

    if not v3_pass:
        raise RuntimeError(
            "V3 production inference does not "
            "match frozen evaluation inference."
        )

    print()
    print("=" * 70)
    print(
        "PRODUCTION INFERENCE PARITY STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
