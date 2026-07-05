from __future__ import annotations

import math

import cv2
import httpx
import numpy as np
import pandas as pd

from backend.app.core.config import PROJECT_ROOT


BASE_URL = "http://127.0.0.1:8000"

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
    / "PNEUMONIA"
    / "person155_bacteria_730.jpeg"
)

V1_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_test_predictions.csv"
)

V3_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_test_v3_predictions.csv"
)

V1_ATOL = 1e-6
V3_ATOL = 2e-4

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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
        "PRODUCTION API INTEGRATION TEST"
    )
    print("=" * 70)

    require(
        TEST_IMAGE.is_file(),
        f"Missing test image: {TEST_IMAGE}",
    )

    v1_predictions = pd.read_csv(
        V1_PREDICTIONS_FILE
    )

    v3_predictions = pd.read_csv(
        V3_PREDICTIONS_FILE
    )

    expected_v1 = float(
        v1_predictions.iloc[0]["probability"]
    )

    expected_v3 = float(
        v3_predictions.iloc[0]["probability"]
    )

    image_bytes = TEST_IMAGE.read_bytes()

    original_image = cv2.imdecode(
        np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        ),
        cv2.IMREAD_UNCHANGED,
    )

    require(
        original_image is not None,
        "Could not decode local test image.",
    )

    original_height = int(
        original_image.shape[0]
    )

    original_width = int(
        original_image.shape[1]
    )

    with httpx.Client(
        base_url=BASE_URL,
        timeout=120.0,
    ) as client:
        print("\nTesting root endpoint...")

        response = client.get("/")

        require(
            response.status_code == 200,
            "Root endpoint failed.",
        )

        root = response.json()

        require(
            root["status"] == "running",
            "Unexpected root status.",
        )

        print("Root endpoint: PASS")

        print("\nTesting health endpoint...")

        response = client.get(
            "/api/v1/health"
        )

        require(
            response.status_code == 200,
            "Health endpoint failed.",
        )

        health = response.json()

        require(
            health["status"] == "ready",
            "API is not ready.",
        )

        require(
            health["models_loaded"] is True,
            "Models are not loaded.",
        )

        print("Health endpoint: PASS")

        print("\nTesting model-info endpoint...")

        response = client.get(
            "/api/v1/model-info"
        )

        require(
            response.status_code == 200,
            "Model-info endpoint failed.",
        )

        model_info = response.json()

        require(
            model_info[
                "primary_model"
            ]["name"]
            == "baseline_cnn_v1",
            "Unexpected primary model.",
        )

        require(
            model_info[
                "deployment_policy"
            ]["automatic_override_allowed"]
            is False,
            "V3 override must remain disabled.",
        )

        require(
            model_info[
                "deployment_policy"
            ]["automatic_ensemble_allowed"]
            is False,
            "Automatic ensemble must remain disabled.",
        )

        print("Model-info endpoint: PASS")

        print("\nTesting valid prediction...")

        response = client.post(
            "/api/v1/predict",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_bytes,
                    "image/jpeg",
                )
            },
        )

        require(
            response.status_code == 200,
            (
                "Valid prediction failed: "
                f"{response.text}"
            ),
        )

        prediction = response.json()

        actual_v1 = float(
            prediction[
                "primary_prediction"
            ]["probability"]
        )

        actual_v3 = float(
            prediction[
                "secondary_signal"
            ]["probability"]
        )

        v1_difference = abs(
            actual_v1 - expected_v1
        )

        v3_difference = abs(
            actual_v3 - expected_v3
        )

        print(
            f"Expected V1:   {expected_v1:.12f}"
        )
        print(
            f"API V1:        {actual_v1:.12f}"
        )
        print(
            f"V1 difference: {v1_difference:.12e}"
        )

        print(
            f"Expected V3:   {expected_v3:.12f}"
        )
        print(
            f"API V3:        {actual_v3:.12f}"
        )
        print(
            f"V3 difference: {v3_difference:.12e}"
        )

        require(
            math.isclose(
                actual_v1,
                expected_v1,
                rel_tol=0.0,
                abs_tol=V1_ATOL,
            ),
            "V1 API probability parity failed.",
        )

        v3_threshold = float(
            prediction[
                "secondary_signal"
            ]["threshold"]
        )

        expected_v3_positive = (
            expected_v3 >= v3_threshold
        )

        actual_v3_positive = (
            actual_v3 >= v3_threshold
        )

        require(
            actual_v3_positive
            == expected_v3_positive,
            (
                "V3 safety-signal decision "
                "parity failed."
            ),
        )

        require(
            prediction[
                "secondary_signal"
            ]["predicted_label"]
            == (
                "PNEUMONIA"
                if actual_v3_positive
                else "NORMAL"
            ),
            (
                "V3 safety-signal label is "
                "inconsistent with threshold."
            ),
        )

        require(
            prediction[
                "decision"
            ]["final_label"]
            == "PNEUMONIA",
            "Unexpected final label.",
        )

        require(
            prediction[
                "decision"
            ]["source"]
            == "baseline_cnn_v1",
            "Unexpected decision source.",
        )

        require(
            prediction[
                "decision"
            ][
                "manual_review_recommended"
            ]
            is False,
            "Unexpected manual-review warning.",
        )

        require(
            prediction[
                "secondary_signal"
            ][
                "automatic_override_allowed"
            ]
            is False,
            "V3 override policy changed.",
        )

        print("Valid prediction: PASS")
        print("V1 probability parity: PASS")
        print("V3 safety-decision parity: PASS")

        print("\nTesting valid explanation...")

        response = client.post(
            "/api/v1/explain",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_bytes,
                    "image/jpeg",
                )
            },
        )

        require(
            response.status_code == 200,
            (
                "Valid explanation failed: "
                f"{response.text}"
            ),
        )

        require(
            response.headers[
                "content-type"
            ].startswith("image/png"),
            "Explanation is not a PNG response.",
        )

        require(
            response.content.startswith(
                PNG_SIGNATURE
            ),
            "Explanation has invalid PNG signature.",
        )

        require(
            response.headers.get(
                "x-mediscan-model"
            )
            == "baseline_cnn_v1",
            "Unexpected explanation model.",
        )

        require(
            response.headers.get(
                "x-mediscan-label"
            )
            == "PNEUMONIA",
            "Unexpected explanation label.",
        )

        explanation_probability = float(
            response.headers[
                "x-mediscan-probability"
            ]
        )

        explanation_difference = abs(
            explanation_probability
            - expected_v1
        )

        print(
            "Expected explanation V1: "
            f"{expected_v1:.12f}"
        )

        print(
            "API explanation V1:      "
            f"{explanation_probability:.12f}"
        )

        print(
            "Explanation difference:  "
            f"{explanation_difference:.12e}"
        )

        require(
            math.isclose(
                explanation_probability,
                expected_v1,
                rel_tol=0.0,
                abs_tol=V1_ATOL,
            ),
            (
                "Explanation probability "
                "parity failed."
            ),
        )

        require(
            response.headers.get(
                "x-mediscan-explanation-method"
            )
            == "gradcam",
            "Unexpected explanation method.",
        )

        explanation_mode = (
            response.headers.get(
                "x-mediscan-explanation-mode"
            )
        )

        require(
            explanation_mode
            in {
                "positive_gradcam",
                "absolute_attribution",
            },
            "Unexpected explanation mode.",
        )

        require(
            response.headers.get(
                "x-mediscan-conv-layer"
            )
            == "conv2d_3",
            "Unexpected Grad-CAM layer.",
        )

        heatmap_array = np.frombuffer(
            response.content,
            dtype=np.uint8,
        )

        decoded_heatmap = cv2.imdecode(
            heatmap_array,
            cv2.IMREAD_UNCHANGED,
        )

        require(
            decoded_heatmap is not None,
            "Could not decode explanation PNG.",
        )

        require(
            decoded_heatmap.ndim == 2,
            (
                "Expected grayscale Grad-CAM "
                "heatmap."
            ),
        )

        require(
            decoded_heatmap.shape
            == (
                original_height,
                original_width,
            ),
            (
                "Explanation dimensions do not "
                "match original image."
            ),
        )

        require(
            int(decoded_heatmap.max()) > 0,
            "Explanation heatmap is empty.",
        )

        print(
            "Explanation HTTP response: PASS"
        )
        print(
            "Explanation PNG signature: PASS"
        )
        print(
            "Explanation dimensions: "
            f"{original_width}x{original_height} PASS"
        )
        print(
            "Explanation probability parity: PASS"
        )
        print(
            "Explanation metadata headers: PASS"
        )

        print(
            "\nTesting explanation quality "
            "headers..."
        )

        quality_header_names = [
            "x-mediscan-quality-status",
            "x-mediscan-display-warning",
            "x-mediscan-warning-code",
            "x-mediscan-border-energy",
            "x-mediscan-thorax-energy",
            "x-mediscan-peak-in-border",
        ]

        heatmap_quality_headers = {
            name: response.headers.get(name)
            for name in quality_header_names
        }

        for name, value in (
            heatmap_quality_headers.items()
        ):
            require(
                value is not None,
                (
                    "Missing explanation quality "
                    f"header: {name}"
                ),
            )

        require(
            heatmap_quality_headers[
                "x-mediscan-quality-status"
            ]
            in {
                "HIGH_SHORTCUT_RISK",
                "ELEVATED_SHORTCUT_RISK",
                "LIMITED_SPATIAL_RELIABILITY",
            },
            "Unexpected explanation quality status.",
        )

        require(
            heatmap_quality_headers[
                "x-mediscan-display-warning"
            ]
            in {"true", "false"},
            "Invalid display-warning header.",
        )

        border_energy = float(
            heatmap_quality_headers[
                "x-mediscan-border-energy"
            ]
        )

        thorax_energy = float(
            heatmap_quality_headers[
                "x-mediscan-thorax-energy"
            ]
        )

        require(
            0.0 <= border_energy <= 1.0,
            "Border energy is outside [0, 1].",
        )

        require(
            0.0 <= thorax_energy <= 1.0,
            "Thorax energy is outside [0, 1].",
        )

        require(
            heatmap_quality_headers[
                "x-mediscan-peak-in-border"
            ]
            in {"true", "false"},
            "Invalid peak-in-border header.",
        )

        print(
            "Explanation quality headers: PASS"
        )

        print(
            "\nTesting valid explanation overlay..."
        )

        overlay_response = client.post(
            "/api/v1/explain/overlay",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_bytes,
                    "image/jpeg",
                )
            },
        )

        require(
            overlay_response.status_code == 200,
            (
                "Valid explanation overlay failed: "
                f"{overlay_response.text}"
            ),
        )

        require(
            overlay_response.headers[
                "content-type"
            ].startswith("image/png"),
            "Overlay is not a PNG response.",
        )

        require(
            overlay_response.content.startswith(
                PNG_SIGNATURE
            ),
            "Overlay has invalid PNG signature.",
        )

        overlay_array = np.frombuffer(
            overlay_response.content,
            dtype=np.uint8,
        )

        decoded_overlay = cv2.imdecode(
            overlay_array,
            cv2.IMREAD_UNCHANGED,
        )

        require(
            decoded_overlay is not None,
            "Could not decode overlay PNG.",
        )

        require(
            (
                decoded_overlay.ndim == 3
                and decoded_overlay.shape[2] == 3
            ),
            "Expected three-channel RGB overlay.",
        )

        require(
            decoded_overlay.shape[:2]
            == (
                original_height,
                original_width,
            ),
            (
                "Overlay dimensions do not match "
                "original image."
            ),
        )

        require(
            overlay_response.headers.get(
                "x-mediscan-visualization"
            )
            == "colored_overlay",
            "Unexpected overlay visualization.",
        )

        require(
            overlay_response.headers.get(
                "x-mediscan-colormap"
            )
            == "jet",
            "Unexpected overlay colormap.",
        )

        require(
            math.isclose(
                float(
                    overlay_response.headers[
                        "x-mediscan-overlay-alpha"
                    ]
                ),
                0.45,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "Unexpected overlay alpha.",
        )

        overlay_quality_headers = {
            name: overlay_response.headers.get(name)
            for name in quality_header_names
        }

        require(
            overlay_quality_headers
            == heatmap_quality_headers,
            (
                "Heatmap and overlay quality "
                "headers do not match."
            ),
        )

        require(
            overlay_response.headers.get(
                "x-mediscan-model"
            )
            == response.headers.get(
                "x-mediscan-model"
            ),
            (
                "Heatmap and overlay model "
                "metadata do not match."
            ),
        )

        require(
            overlay_response.headers.get(
                "x-mediscan-label"
            )
            == response.headers.get(
                "x-mediscan-label"
            ),
            (
                "Heatmap and overlay labels "
                "do not match."
            ),
        )

        require(
            overlay_response.headers.get(
                "x-mediscan-explanation-mode"
            )
            == response.headers.get(
                "x-mediscan-explanation-mode"
            ),
            (
                "Heatmap and overlay explanation "
                "modes do not match."
            ),
        )

        print(
            "Overlay HTTP response: PASS"
        )
        print(
            "Overlay PNG signature: PASS"
        )
        print(
            "Overlay RGB dimensions: "
            f"{original_width}x{original_height} PASS"
        )
        print(
            "Overlay visualization metadata: PASS"
        )
        print(
            "Heatmap/overlay quality consistency: PASS"
        )

        print(
            "\nTesting combined analysis endpoint..."
        )

        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_bytes,
                    "image/jpeg",
                )
            },
        )

        require(
            response.status_code == 200,
            (
                "Valid combined analysis failed: "
                f"{response.text}"
            ),
        )

        analysis = response.json()

        required_analysis_keys = {
            "primary_prediction",
            "secondary_signal",
            "decision",
            "preprocessing",
            "explanation",
            "explanation_quality",
            "visualization_endpoints",
            "disclaimer",
        }

        require(
            set(analysis.keys())
            == required_analysis_keys,
            (
                "Unexpected combined analysis "
                "response schema."
            ),
        )

        analysis_v1 = float(
            analysis[
                "primary_prediction"
            ]["probability"]
        )

        analysis_v3 = float(
            analysis[
                "secondary_signal"
            ]["probability"]
        )

        require(
            math.isclose(
                analysis_v1,
                expected_v1,
                rel_tol=0.0,
                abs_tol=V1_ATOL,
            ),
            (
                "Combined analysis V1 "
                "probability parity failed."
            ),
        )

        require(
            math.isclose(
                analysis_v3,
                expected_v3,
                rel_tol=0.0,
                abs_tol=V3_ATOL,
            ),
            (
                "Combined analysis V3 "
                "probability parity failed."
            ),
        )

        require(
            analysis[
                "decision"
            ]["source"]
            == "baseline_cnn_v1",
            (
                "Combined analysis changed "
                "the authoritative source."
            ),
        )

        require(
            analysis[
                "secondary_signal"
            ][
                "automatic_override_allowed"
            ]
            is False,
            (
                "Combined analysis changed "
                "the V3 override policy."
            ),
        )

        analysis_explanation = analysis[
            "explanation"
        ]

        analysis_quality = analysis[
            "explanation_quality"
        ]

        require(
            analysis_explanation["method"]
            == "gradcam",
            (
                "Unexpected combined analysis "
                "explanation method."
            ),
        )

        require(
            analysis_explanation["mode"]
            in {
                "positive_gradcam",
                "absolute_attribution",
            },
            (
                "Unexpected combined analysis "
                "explanation mode."
            ),
        )

        require(
            analysis_quality[
                "explanation_mode"
            ]
            == analysis_explanation["mode"],
            (
                "Combined analysis explanation "
                "and quality modes differ."
            ),
        )

        require(
            analysis_quality[
                "quality_status"
            ]
            in {
                "HIGH_SHORTCUT_RISK",
                "ELEVATED_SHORTCUT_RISK",
                "LIMITED_SPATIAL_RELIABILITY",
            },
            (
                "Unexpected combined analysis "
                "quality status."
            ),
        )

        require(
            isinstance(
                analysis_quality[
                    "display_warning"
                ],
                bool,
            ),
            (
                "Combined analysis warning "
                "flag is not boolean."
            ),
        )

        require(
            0.0
            <= float(
                analysis_quality[
                    "border_energy_ratio"
                ]
            )
            <= 1.0,
            (
                "Combined analysis border "
                "energy is outside [0, 1]."
            ),
        )

        require(
            0.0
            <= float(
                analysis_quality[
                    "thorax_energy_ratio"
                ]
            )
            <= 1.0,
            (
                "Combined analysis thorax "
                "energy is outside [0, 1]."
            ),
        )

        endpoints = analysis[
            "visualization_endpoints"
        ]

        require(
            endpoints["heatmap"]
            == "/api/v1/explain",
            (
                "Unexpected combined analysis "
                "heatmap endpoint."
            ),
        )

        require(
            endpoints["overlay"]
            == "/api/v1/explain/overlay",
            (
                "Unexpected combined analysis "
                "overlay endpoint."
            ),
        )

        print(
            f"Analysis V1:          "
            f"{analysis_v1:.12f}"
        )
        print(
            f"Analysis V3:          "
            f"{analysis_v3:.12f}"
        )
        print(
            f"Explanation mode:     "
            f"{analysis_explanation['mode']}"
        )
        print(
            f"Quality status:       "
            f"{analysis_quality['quality_status']}"
        )
        print(
            f"Display warning:      "
            f"{analysis_quality['display_warning']}"
        )

        print(
            "Combined analysis HTTP response: PASS"
        )
        print(
            "Combined analysis schema: PASS"
        )
        print(
            "Combined analysis probability parity: PASS"
        )
        print(
            "Combined analysis deployment policy: PASS"
        )
        print(
            "Combined analysis explanation metadata: PASS"
        )
        print(
            "Combined analysis quality metadata: PASS"
        )
        print(
            "Combined analysis visualization contract: PASS"
        )

        print(
            "\nTesting analysis unsupported "
            "content type..."
        )

        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "invalid.txt",
                    b"not an image",
                    "text/plain",
                )
            },
        )

        require(
            response.status_code == 415,
            (
                "Expected 415 for analysis "
                "unsupported content type."
            ),
        )

        print(
            "Analysis unsupported content "
            "type: PASS"
        )

        print(
            "\nTesting analysis fake JPEG..."
        )

        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "fake.jpeg",
                    b"not an image",
                    "image/jpeg",
                )
            },
        )

        require(
            response.status_code == 400,
            (
                "Expected 400 for analysis "
                "fake JPEG."
            ),
        )

        print(
            "Analysis fake JPEG rejection: PASS"
        )

        print(
            "\nTesting prediction unsupported "
            "content type..."
        )

        response = client.post(
            "/api/v1/predict",
            files={
                "file": (
                    "invalid.txt",
                    b"not an image",
                    "text/plain",
                )
            },
        )

        require(
            response.status_code == 415,
            (
                "Expected 415 for prediction "
                "unsupported content type."
            ),
        )

        print(
            "Prediction unsupported content "
            "type: PASS"
        )

        print(
            "\nTesting prediction fake JPEG..."
        )

        response = client.post(
            "/api/v1/predict",
            files={
                "file": (
                    "fake.jpeg",
                    b"not an image",
                    "image/jpeg",
                )
            },
        )

        require(
            response.status_code == 400,
            (
                "Expected 400 for prediction "
                "fake JPEG."
            ),
        )

        print(
            "Prediction fake JPEG rejection: PASS"
        )

        print(
            "\nTesting explanation unsupported "
            "content type..."
        )

        response = client.post(
            "/api/v1/explain",
            files={
                "file": (
                    "invalid.txt",
                    b"not an image",
                    "text/plain",
                )
            },
        )

        require(
            response.status_code == 415,
            (
                "Expected 415 for explanation "
                "unsupported content type."
            ),
        )

        print(
            "Explanation unsupported content "
            "type: PASS"
        )

        print(
            "\nTesting explanation fake JPEG..."
        )

        response = client.post(
            "/api/v1/explain",
            files={
                "file": (
                    "fake.jpeg",
                    b"not an image",
                    "image/jpeg",
                )
            },
        )

        require(
            response.status_code == 400,
            (
                "Expected 400 for explanation "
                "fake JPEG."
            ),
        )

        print(
            "Explanation fake JPEG rejection: PASS"
        )

    print("\n" + "=" * 70)
    print(
        "PRODUCTION API INTEGRATION "
        "TEST STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
