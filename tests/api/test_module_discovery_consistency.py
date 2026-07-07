from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.modules.registry import (
    MEDICAL_MODULES,
)
from backend.app.modules.responses import (
    ModuleDiscoveryItem,
    module_discovery_item_from_registry,
)


pytestmark = pytest.mark.api


@pytest.mark.parametrize(
    "module",
    MEDICAL_MODULES,
)
def test_registry_module_projects_to_exact_discovery_item(
    module,
) -> None:
    item = module_discovery_item_from_registry(module)

    assert isinstance(item, ModuleDiscoveryItem)
    assert item.module_id == module.module_id
    assert item.display_name == module.display_name
    assert item.modality is module.modality
    assert item.task_type is module.task_type
    assert item.status is module.status
    assert item.output_classes == module.output_classes
    assert (
        item.supports_gradcam
        is module.supports_gradcam
    )
    assert item.executable is module.executable


def test_discovery_api_matches_registry_exactly(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/modules")

    assert response.status_code == 200

    expected = [
        module_discovery_item_from_registry(
            module
        ).model_dump(mode="json")
        for module in MEDICAL_MODULES
    ]

    assert response.json() == {
        "modules": expected,
        "total": len(MEDICAL_MODULES),
    }
