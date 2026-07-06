from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from backend.app.modules.registry import (
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
