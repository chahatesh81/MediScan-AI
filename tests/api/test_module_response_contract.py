from __future__ import annotations

from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from backend.app.services.analysis_service import (
    analysis_service,
)


pytestmark = pytest.mark.api


def make_pneumonia_payload() -> dict[str, object]:
    return {
        "primary_prediction": {
            "model": "baseline_cnn_v1",
            "label": "PNEUMONIA",
            "probability": 0.81,
            "threshold": 0.53,
        },
        "secondary_signal": {
            "model": "advanced_v3",
        },
    }


def test_module_analysis_returns_generic_response_contract(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze_mock = Mock(
        return_value=make_pneumonia_payload()
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
    assert response.json() == {
        "module_id": "pneumonia_detection",
        "display_name": "Pneumonia Detection",
        "modality": "chest_xray",
        "task_type": "binary_classification",
        "result": {
            "task_type": "binary_classification",
            "negative_label": "NORMAL",
            "positive_label": "PNEUMONIA",
            "predicted_label": "PNEUMONIA",
            "probability": 0.81,
            "threshold": 0.53,
        },
    }


def test_module_response_excludes_legacy_analysis_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analysis_service,
        "analyze_bytes",
        Mock(
            return_value=make_pneumonia_payload()
        ),
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

    payload = response.json()

    assert response.status_code == 200
    assert "primary_prediction" not in payload
    assert "secondary_signal" not in payload
    assert "decision" not in payload
    assert "preprocessing" not in payload
    assert "explanation" not in payload


def test_module_response_preserves_executor_call(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze_mock = Mock(
        return_value=make_pneumonia_payload()
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
