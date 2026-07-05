from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.services.explanation_service import (
    explanation_service,
)


pytestmark = pytest.mark.api


def explanation_result(
    payload_key: str,
) -> dict[str, object]:
    return {
        "model": "baseline_cnn_v1",
        "prediction": {
            "label": "PNEUMONIA",
            "probability": 0.91,
            "threshold": 0.53,
        },
        "explanation": {
            "method": "gradcam",
            "mode": "positive_gradcam",
            "last_conv_layer": "conv2d_2",
            "visualization": "overlay",
            "colormap": "jet",
            "overlay_alpha": 0.45,
        },
        "quality": {
            "quality_status": (
                "LIMITED_SPATIAL_RELIABILITY"
            ),
            "display_warning": False,
            "warning_code": None,
            "border_energy_ratio": 0.12,
            "thorax_energy_ratio": 0.76,
            "peak_in_border": False,
        },
        payload_key: b"\x89PNG\r\n\x1a\nfake-png",
    }


def test_explain_returns_png_and_metadata_headers(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        explanation_service,
        "explain_bytes",
        lambda image_bytes: explanation_result(
            "heatmap_png_bytes"
        ),
    )

    response = client.post(
        "/api/v1/explain",
        files={
            "file": (
                "xray.png",
                b"image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-mediscan-model"] == (
        "baseline_cnn_v1"
    )
    assert response.headers["x-mediscan-label"] == (
        "PNEUMONIA"
    )
    assert response.headers[
        "x-mediscan-explanation-method"
    ] == "gradcam"
    assert response.headers[
        "x-mediscan-display-warning"
    ] == "false"
    assert response.content.startswith(b"\x89PNG")


def test_explain_overlay_returns_png(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        explanation_service,
        "overlay_bytes",
        lambda image_bytes: explanation_result(
            "overlay_png_bytes"
        ),
    )

    response = client.post(
        "/api/v1/explain/overlay",
        files={
            "file": (
                "xray.png",
                b"image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


@pytest.mark.parametrize(
    ("endpoint", "method_name", "message"),
    [
        (
            "/api/v1/explain",
            "explain_bytes",
            "Explanation could not be generated.",
        ),
        (
            "/api/v1/explain/overlay",
            "overlay_bytes",
            (
                "Explanation overlay could "
                "not be generated."
            ),
        ),
    ],
)
def test_explanation_endpoints_hide_internal_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    method_name: str,
    message: str,
) -> None:
    def fail(image_bytes: bytes) -> None:
        raise RuntimeError("private explanation failure")

    monkeypatch.setattr(
        explanation_service,
        method_name,
        fail,
    )

    response = client.post(
        endpoint,
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
        "detail": message
    }
    assert "private explanation failure" not in response.text
