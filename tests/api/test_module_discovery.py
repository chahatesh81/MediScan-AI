from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.modules.registry import (
    MEDICAL_MODULES,
)


pytestmark = pytest.mark.api


def test_module_discovery_returns_all_registered_modules(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/modules")

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 5
    assert len(body["modules"]) == 5


def test_module_discovery_preserves_registry_order(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/modules")

    module_ids = [
        module["module_id"]
        for module in response.json()["modules"]
    ]

    assert module_ids == [
        module.module_id
        for module in MEDICAL_MODULES
    ]


def test_module_discovery_exposes_complete_metadata(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/modules")

    first = response.json()["modules"][0]

    assert set(first) == {
        "module_id",
        "display_name",
        "modality",
        "task_type",
        "status",
        "output_classes",
        "supports_gradcam",
        "executable",
    }


def test_module_discovery_exposes_pneumonia_as_available(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/modules")

    modules = {
        module["module_id"]: module
        for module in response.json()["modules"]
    }

    pneumonia = modules["pneumonia_detection"]

    assert pneumonia["status"] == "AVAILABLE"
    assert pneumonia["executable"] is True
    assert pneumonia["task_type"] == (
        "binary_classification"
    )
    assert pneumonia["output_classes"] == [
        "NORMAL",
        "PNEUMONIA",
    ]
    assert pneumonia["supports_gradcam"] is True


@pytest.mark.parametrize(
    "module_id",
    [
        "brain_tumor_mri",
        "skin_disease",
        "chest_multidisease",
        "breast_cancer",
    ],
)
def test_module_discovery_exposes_planned_modules_as_non_executable(
    client: TestClient,
    module_id: str,
) -> None:
    response = client.get("/api/v1/modules")

    modules = {
        module["module_id"]: module
        for module in response.json()["modules"]
    }

    module = modules[module_id]

    assert module["status"] == "PLANNED"
    assert module["executable"] is False


def test_module_discovery_serializes_enum_values(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/modules")

    for module in response.json()["modules"]:
        assert isinstance(module["modality"], str)
        assert isinstance(module["task_type"], str)
        assert isinstance(module["status"], str)


def test_module_discovery_output_classes_are_json_arrays(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/modules")

    for module in response.json()["modules"]:
        assert isinstance(
            module["output_classes"],
            list,
        )
        assert len(module["output_classes"]) >= 2
