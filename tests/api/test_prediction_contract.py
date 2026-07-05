from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.services.inference_service import (
    inference_service,
)


pytestmark = pytest.mark.api


def prediction_result() -> dict[str, object]:
    return {
        "primary_prediction": {
            "model": "baseline_cnn_v1",
            "label": "PNEUMONIA",
            "probability": 0.91,
            "threshold": 0.53,
        },
        "secondary_signal": {
            "model": "advanced_v3",
            "role": "exploratory",
            "probability": 0.72,
            "threshold": 0.50,
            "predicted_label": "PNEUMONIA",
            "automatic_override_allowed": False,
        },
        "decision": {
            "final_label": "PNEUMONIA",
            "source": "baseline_cnn_v1",
            "manual_review_recommended": False,
            "warning_code": None,
        },
        "preprocessing": {
            "v1": "rgb_bilinear_resize_224",
            "v3": "artifact_aware_preprocess_xray",
            "v3_metadata": {
                "original_width": 512,
                "original_height": 512,
                "crop_bbox": {
                    "x_min": 10,
                    "y_min": 20,
                    "x_max": 500,
                    "y_max": 490,
                },
                "cropped_width": 490,
                "cropped_height": 470,
                "retained_area_ratio": 0.88,
                "target_size": 224,
            },
        },
        "disclaimer": (
            "Educational decision-support prototype. "
            "Not for clinical use. Human review required."
        ),
    }


def test_predict_returns_structured_prediction(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference_service,
        "predict_bytes",
        lambda image_bytes: prediction_result(),
    )

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "xray.jpg",
                b"valid-test-bytes",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["primary_prediction"]["model"] == (
        "baseline_cnn_v1"
    )
    assert body["primary_prediction"]["label"] == (
        "PNEUMONIA"
    )
    assert body["decision"]["final_label"] == (
        "PNEUMONIA"
    )
    assert (
        body["secondary_signal"][
            "automatic_override_allowed"
        ]
        is False
    )


def test_predict_maps_value_error_to_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(image_bytes: bytes) -> None:
        raise ValueError("Invalid image payload.")

    monkeypatch.setattr(
        inference_service,
        "predict_bytes",
        fail,
    )

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "xray.jpg",
                b"invalid-image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid image payload."
    }


def test_predict_hides_internal_server_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(image_bytes: bytes) -> None:
        raise RuntimeError("secret internal failure")

    monkeypatch.setattr(
        inference_service,
        "predict_bytes",
        fail,
    )

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "xray.jpg",
                b"image-bytes",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "Prediction could not be completed."
        )
    }

    assert "secret internal failure" not in response.text
