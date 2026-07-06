from __future__ import annotations

from fastapi import APIRouter

from backend.app.modules.runtime_validation import (
    validate_module_runtime,
)
from backend.app.modules.responses import (
    ModuleRuntimeHealthResponse,
)
from backend.app.modules.registry import (
    MedicalModule,
    list_modules,
)


router = APIRouter()


def serialize_module(
    module: MedicalModule,
) -> dict[str, object]:
    return {
        "module_id": module.module_id,
        "display_name": module.display_name,
        "modality": module.modality.value,
        "task_type": module.task_type.value,
        "status": module.status.value,
        "output_classes": list(module.output_classes),
        "supports_gradcam": module.supports_gradcam,
        "executable": module.executable,
    }


@router.get(
    "/modules",
    tags=["Modules"],
)
def discover_modules() -> dict[str, object]:
    modules = list_modules()

    return {
        "modules": [
            serialize_module(module)
            for module in modules
        ],
        "total": len(modules),
    }


@router.get(
    "/modules/runtime",
    response_model=ModuleRuntimeHealthResponse,
    tags=["Modules"],
)
def module_runtime_health(
) -> ModuleRuntimeHealthResponse:
    result = validate_module_runtime()

    return ModuleRuntimeHealthResponse(
        status="ready",
        validated_module_ids=(
            result.validated_module_ids
        ),
        validated_module_count=len(
            result.validated_module_ids
        ),
    )
