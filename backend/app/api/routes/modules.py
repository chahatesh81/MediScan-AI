from __future__ import annotations

from fastapi import APIRouter

from backend.app.modules.runtime_validation import (
    validate_module_runtime,
)
from backend.app.modules.responses import (
    ModuleDiscoveryResponse,
    ModuleRuntimeHealthResponse,
    module_discovery_item_from_registry,
    module_runtime_health_response,
)
from backend.app.modules.registry import (
    list_modules,
)


router = APIRouter()


@router.get(
    "/modules",
    response_model=ModuleDiscoveryResponse,
    tags=["Modules"],
)
def discover_modules() -> ModuleDiscoveryResponse:
    modules = list_modules()

    return ModuleDiscoveryResponse(
        modules=tuple(
            module_discovery_item_from_registry(module)
            for module in modules
        ),
        total=len(modules),
    )


@router.get(
    "/modules/runtime",
    response_model=ModuleRuntimeHealthResponse,
    tags=["Modules"],
)
def module_runtime_health(
) -> ModuleRuntimeHealthResponse:
    result = validate_module_runtime()

    return module_runtime_health_response(
        result.validated_module_ids
    )
