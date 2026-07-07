from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes.modules import (
    discover_modules,
    serialize_module,
)
from backend.app.modules.registry import (
    PNEUMONIA_DETECTION,
)
from backend.app.modules.responses import (
    ModuleDiscoveryItem,
    ModuleDiscoveryResponse,
)


pytestmark = pytest.mark.api


def test_module_serializer_returns_typed_contract(
) -> None:
    result = serialize_module(
        PNEUMONIA_DETECTION
    )

    assert isinstance(
        result,
        ModuleDiscoveryItem,
    )
    assert result.module_id == (
        "pneumonia_detection"
    )
    assert result.output_classes == (
        "NORMAL",
        "PNEUMONIA",
    )


def test_discovery_function_returns_response_contract(
) -> None:
    result = discover_modules()

    assert isinstance(
        result,
        ModuleDiscoveryResponse,
    )
    assert result.total == len(result.modules)
    assert result.total == 5


def test_module_discovery_api_contract_is_exact(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/modules"
    )

    assert response.status_code == 200

    payload = response.json()

    assert set(payload) == {
        "modules",
        "total",
    }

    assert set(payload["modules"][0]) == {
        "module_id",
        "display_name",
        "modality",
        "task_type",
        "status",
        "output_classes",
        "supports_gradcam",
        "executable",
    }


def test_module_discovery_contract_preserves_json_shape(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/modules"
    )

    module = response.json()["modules"][0]

    assert module["module_id"] == (
        "pneumonia_detection"
    )
    assert module["output_classes"] == [
        "NORMAL",
        "PNEUMONIA",
    ]
    assert module["status"] == "AVAILABLE"
    assert module["executable"] is True
