from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from backend.app.modules.registry import (
    ModuleStatus,
    MedicalModality,
    TaskType,
)
from backend.app.modules.results import (
    BinaryClassificationResult,
    MulticlassClassificationResult,
    MultilabelClassificationResult,
)


ModuleTaskResult = (
    BinaryClassificationResult
    | MulticlassClassificationResult
    | MultilabelClassificationResult
)


class ModuleAnalysisResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    module_id: str
    display_name: str
    modality: MedicalModality
    task_type: TaskType
    result: ModuleTaskResult


class ModuleRuntimeHealthResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: str
    validated_module_ids: tuple[str, ...]
    validated_module_count: int


def module_discovery_item_from_registry(
    module: MedicalModule,
) -> "ModuleDiscoveryItem":
    return ModuleDiscoveryItem(
        module_id=module.module_id,
        display_name=module.display_name,
        modality=module.modality,
        task_type=module.task_type,
        status=module.status,
        output_classes=module.output_classes,
        supports_gradcam=module.supports_gradcam,
        executable=module.executable,
    )


class ModuleDiscoveryItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    module_id: str
    display_name: str
    modality: MedicalModality
    task_type: TaskType
    status: ModuleStatus
    output_classes: tuple[str, ...]
    supports_gradcam: bool
    executable: bool


class ModuleDiscoveryResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    modules: tuple[ModuleDiscoveryItem, ...]
    total: int
