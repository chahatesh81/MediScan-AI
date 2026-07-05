from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.services.analysis_service import (
    analysis_service,
)


pytestmark = pytest.mark.api


def analysis_result() -> dict[str, object]:
    return {
        "primary_prediction": {
            "model": "baseline_cnn_v1",
            "label": "NORMAL",
            "probability": 0.20,
            "threshold": 0.53,
        },
        "secondary_signal": {
            "model": "advanced_v3",
            "role": "exploratory",
            "probability": 0.80,
            "threshold": 0.50,
            "predicted_label": "PNEUMONIA",
            "automatic_override_allowed": False,
        },
        "decision": {
            "final_label": "NORMAL",
            "source": "baseline_cnn_v1",
            "manual_review_recommended": True,
            "warning_code": (
                "V1_NORMAL_V3_PNEUMONIA"
            ),
        },
        "preprocessing": {
            "v1": "rgb_bilinear_resize_224",
            "v3": "artifact_aware_preprocess_xray",
            "v3_metadata": {
                "original_width": 512,
                "original_height": 512,
                "crop_bbox": {
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 512,
                    "y_max": 512,
                },
                "cropped_width": 512,
                "cropped_height": 512,
                "retained_area_ratio": 1.0,
                "target_size": 224,
            },
        },
        "explanation": {
            "method": "gradcam",
            "mode": "positive_gradcam",
            "last_conv_layer": "conv2d_2",
            "raw_heatmap_shape": [14, 14],
            "output_width": 512,
            "output_height": 512,
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "explanation_quality": {
            "border_energy_ratio": 0.12,
            "thorax_energy_ratio": 0.76,
            "peak_in_border": 0.0,
            "quality_status": (
                "LIMITED_SPATIAL_RELIABILITY"
            ),
            "display_warning": False,
            "warning_code": None,
            "explanation_mode": "positive_gradcam",
            "attribution_note": None,
            "region_definition": "test-region",
        },
        "visualization_endpoints": {
            "heatmap": "/api/v1/explain",
            "overlay": "/api/v1/explain/overlay",
        },
        "disclaimer": (
            "Educational decision-support prototype. "
            "Not for clinical use. Human review required."
        ),
    }


def test_analyze_returns_combined_contract(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        lambda image_bytes: analysis_result(),
    )

    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "xray.png",
                b"valid-test-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["decision"]["final_label"] == "NORMAL"
    assert (
        body["decision"][
            "manual_review_recommended"
        ]
        is True
    )
    assert body["decision"]["warning_code"] == (
        "V1_NORMAL_V3_PNEUMONIA"
    )
    assert body["explanation"]["method"] == "gradcam"
    assert body["visualization_endpoints"] == {
        "heatmap": "/api/v1/explain",
        "overlay": "/api/v1/explain/overlay",
    }


def test_analyze_maps_value_error_to_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(image_bytes: bytes) -> None:
        raise ValueError("Analysis image is invalid.")

    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        fail,
    )

    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "xray.png",
                b"invalid-image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Analysis image is invalid."
    }


def test_analyze_hides_internal_server_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(image_bytes: bytes) -> None:
        raise RuntimeError("private analysis failure")

    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        fail,
    )

    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "xray.png",
                b"image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "Combined analysis could "
            "not be completed."
        )
    }

    assert "private analysis failure" not in response.text
