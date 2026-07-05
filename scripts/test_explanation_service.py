from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT
from backend.app.services.explanation_service import (
    explanation_service,
)


TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
    / "PNEUMONIA"
    / "person155_bacteria_730.jpeg"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_test_predictions.csv"
)

V1_ATOL = 1e-6


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "EXPLANATION SERVICE TEST"
    )
    print("=" * 70)

    require(
        TEST_IMAGE.is_file(),
        f"Missing test image: {TEST_IMAGE}",
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    expected_probability = float(
        predictions.iloc[0]["probability"]
    )

    result = (
        explanation_service.explain_bytes(
            TEST_IMAGE.read_bytes()
        )
    )

    actual_probability = float(
        result["prediction"]["probability"]
    )

    difference = abs(
        actual_probability
        - expected_probability
    )

    explanation = result["explanation"]

    print(
        f"Model:            "
        f"{result['model']}"
    )
    print(
        f"Label:            "
        f"{result['prediction']['label']}"
    )
    print(
        f"Expected prob:    "
        f"{expected_probability:.12f}"
    )
    print(
        f"Service prob:     "
        f"{actual_probability:.12f}"
    )
    print(
        f"Difference:       "
        f"{difference:.12e}"
    )
    print(
        f"Mode:             "
        f"{explanation['mode']}"
    )
    print(
        f"Last conv layer:  "
        f"{explanation['last_conv_layer']}"
    )
    print(
        f"Raw heatmap:      "
        f"{explanation['raw_heatmap_shape']}"
    )
    print(
        f"Heatmap range:    "
        f"{explanation['minimum']:.6f} "
        f"to "
        f"{explanation['maximum']:.6f}"
    )
    print(
        f"PNG bytes:        "
        f"{len(result['heatmap_png_bytes'])}"
    )

    require(
        result["model"]
        == "baseline_cnn_v1",
        "Unexpected explanation model.",
    )

    require(
        math.isclose(
            actual_probability,
            expected_probability,
            rel_tol=0.0,
            abs_tol=V1_ATOL,
        ),
        "Explanation prediction parity failed.",
    )

    require(
        result["prediction"]["label"]
        == "PNEUMONIA",
        "Unexpected prediction label.",
    )

    require(
        explanation["mode"]
        in {
            "positive_gradcam",
            "absolute_attribution",
        },
        "Unexpected explanation mode.",
    )

    require(
        explanation["minimum"] >= 0.0,
        "Heatmap minimum is invalid.",
    )

    require(
        explanation["maximum"]
        <= 1.000001,
        "Heatmap maximum is invalid.",
    )

    require(
        explanation["maximum"] > 0.0,
        "Heatmap is numerically zero.",
    )

    png_array = np.frombuffer(
        result["heatmap_png_bytes"],
        dtype=np.uint8,
    )

    decoded_png = cv2.imdecode(
        png_array,
        cv2.IMREAD_GRAYSCALE,
    )

    require(
        decoded_png is not None,
        "Encoded heatmap PNG is invalid.",
    )

    require(
        decoded_png.shape
        == (
            explanation["output_height"],
            explanation["output_width"],
        ),
        (
            "Decoded heatmap dimensions "
            "do not match metadata."
        ),
    )

    print("\nPrediction parity: PASS")
    print("Heatmap validation: PASS")
    print("PNG encoding: PASS")

    print("\n" + "=" * 70)
    print(
        "EXPLANATION SERVICE TEST "
        "STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
