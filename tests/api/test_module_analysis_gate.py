from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.services.analysis_service import (
    analysis_service,
)


pytestmark = pytest.mark.api


def test_unknown_module_returns_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze_mock = Mock()
    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        analyze_mock,
    )

    response = client.post(
        "/api/v1/modules/unknown_module/analyze",
        files={
            "file": (
                "scan.png",
                b"should-not-be-analyzed",
                "image/png",
            )
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            "Unknown medical module: "
            "unknown_module"
        )
    }
    analyze_mock.assert_not_called()


@pytest.mark.parametrize(
    "module_id",
    [
        "brain_tumor_mri",
        "skin_disease",
        "chest_multidisease",
        "breast_cancer",
    ],
)
def test_planned_module_returns_409_without_analysis(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    module_id: str,
) -> None:
    analyze_mock = Mock()
    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        analyze_mock,
    )

    response = client.post(
        f"/api/v1/modules/{module_id}/analyze",
        files={
            "file": (
                "scan.png",
                b"should-not-be-analyzed",
                "image/png",
            )
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            "Medical module is not executable: "
            f"{module_id}"
        )
    }
    analyze_mock.assert_not_called()


def test_pneumonia_module_reaches_analysis_service(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
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
            "peak_in_border": False,
            "quality_status": (
                "LIMITED_SPATIAL_RELIABILITY"
            ),
            "display_warning": False,
            "warning_code": None,
            "explanation_mode": (
                "positive_gradcam"
            ),
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

    analyze_mock = Mock(
        return_value=expected
    )
    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        analyze_mock,
    )

    response = client.post(
        (
            "/api/v1/modules/"
            "pneumonia_detection/analyze"
        ),
        files={
            "file": (
                "scan.png",
                b"valid-image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    analyze_mock.assert_called_once_with(
        b"valid-image-bytes"
    )


def test_executable_module_preserves_upload_validation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze_mock = Mock()
    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        analyze_mock,
    )

    response = client.post(
        (
            "/api/v1/modules/"
            "pneumonia_detection/analyze"
        ),
        files={
            "file": (
                "scan.txt",
                b"not-an-image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    analyze_mock.assert_not_called()


def test_executable_module_rejects_empty_upload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze_mock = Mock()
    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        analyze_mock,
    )

    response = client.post(
        (
            "/api/v1/modules/"
            "pneumonia_detection/analyze"
        ),
        files={
            "file": (
                "scan.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    analyze_mock.assert_not_called()


def test_analysis_value_error_maps_to_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_value_error(
        image_bytes: bytes,
    ) -> None:
        raise ValueError(
            "Module image is invalid."
        )

    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        raise_value_error,
    )

    response = client.post(
        (
            "/api/v1/modules/"
            "pneumonia_detection/analyze"
        ),
        files={
            "file": (
                "scan.png",
                b"valid-image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Module image is invalid."
    }


def test_unexpected_analysis_error_maps_to_safe_500(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_runtime_error(
        image_bytes: bytes,
    ) -> None:
        raise RuntimeError(
            "private module failure"
        )

    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        raise_runtime_error,
    )

    response = client.post(
        (
            "/api/v1/modules/"
            "pneumonia_detection/analyze"
        ),
        files={
            "file": (
                "scan.png",
                b"valid-image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "Module analysis could not "
            "be completed."
        )
    }
    assert (
        "private module failure"
        not in response.text
    )
