from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import modules


pytestmark = pytest.mark.api


def test_module_runtime_health_returns_validated_runtime(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/modules/runtime"
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "validated_module_ids": [
            "pneumonia_detection",
        ],
        "validated_module_count": 1,
    }


def test_module_runtime_health_uses_runtime_validator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = Mock(
        validated_module_ids=(
            "pneumonia_detection",
        )
    )
    validate_mock = Mock(
        return_value=result
    )

    monkeypatch.setattr(
        modules,
        "validate_module_runtime",
        validate_mock,
    )

    response = client.get(
        "/api/v1/modules/runtime"
    )

    assert response.status_code == 200
    validate_mock.assert_called_once_with()


def test_module_runtime_health_response_contract_is_exact(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/modules/runtime"
    )

    assert set(response.json()) == {
        "status",
        "validated_module_ids",
        "validated_module_count",
    }
