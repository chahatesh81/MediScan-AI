from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import modules
from backend.app.modules.responses import (
    ModuleRuntimeHealthResponse,
    module_runtime_health_response,
)


pytestmark = pytest.mark.api


def test_runtime_health_projection_derives_count(
) -> None:
    response = module_runtime_health_response(
        (
            "pneumonia_detection",
            "future_executable_module",
        )
    )

    assert isinstance(
        response,
        ModuleRuntimeHealthResponse,
    )
    assert response.status == "ready"
    assert response.validated_module_ids == (
        "pneumonia_detection",
        "future_executable_module",
    )
    assert response.validated_module_count == 2


def test_runtime_health_projection_accepts_empty_runtime(
) -> None:
    response = module_runtime_health_response(())

    assert response.validated_module_ids == ()
    assert response.validated_module_count == 0


def test_runtime_health_projection_rejects_duplicate_ids(
) -> None:
    with pytest.raises(
        ValueError,
        match="Validated module IDs must be unique",
    ):
        module_runtime_health_response(
            (
                "pneumonia_detection",
                "pneumonia_detection",
            )
        )


def test_runtime_health_route_uses_projection_boundary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = Mock(
        validated_module_ids=(
            "pneumonia_detection",
        )
    )
    projection_mock = Mock(
        return_value=ModuleRuntimeHealthResponse(
            status="ready",
            validated_module_ids=(
                "pneumonia_detection",
            ),
            validated_module_count=1,
        )
    )

    monkeypatch.setattr(
        modules,
        "validate_module_runtime",
        Mock(return_value=result),
    )
    monkeypatch.setattr(
        modules,
        "module_runtime_health_response",
        projection_mock,
    )

    response = client.get(
        "/api/v1/modules/runtime"
    )

    assert response.status_code == 200
    projection_mock.assert_called_once_with(
        result.validated_module_ids
    )


def test_runtime_health_api_count_matches_ids(
    client: TestClient,
) -> None:
    payload = client.get(
        "/api/v1/modules/runtime"
    ).json()

    assert payload["validated_module_count"] == len(
        payload["validated_module_ids"]
    )
    assert len(
        set(payload["validated_module_ids"])
    ) == len(
        payload["validated_module_ids"]
    )
