from __future__ import annotations

from typing import Any

from backend.app.modules.registry import (
    MedicalModule,
    TaskType,
)
from backend.app.modules.results import (
    BinaryClassificationResult,
    MulticlassClassificationResult,
    MultilabelClassificationResult,
)


NormalizedModuleResult = (
    BinaryClassificationResult
    | MulticlassClassificationResult
    | MultilabelClassificationResult
)


class UnsupportedResultNormalizationError(
    RuntimeError
):
    """Raised when no normalizer exists for a module task."""

    def __init__(
        self,
        module: MedicalModule,
    ) -> None:
        self.module_id = module.module_id
        self.task_type = module.task_type
        super().__init__(
            "No result normalizer is available for "
            f"medical module: {module.module_id}"
        )


def normalize_pneumonia_result(
    payload: dict[str, Any],
) -> BinaryClassificationResult:
    primary_prediction = payload[
        "primary_prediction"
    ]

    return BinaryClassificationResult(
        predicted_label=primary_prediction["label"],
        probability=primary_prediction["probability"],
        threshold=primary_prediction["threshold"],
        negative_label="NORMAL",
        positive_label="PNEUMONIA",
    )


def supports_module_result_normalization(
    module: MedicalModule,
) -> bool:
    return module.module_id == "pneumonia_detection"


def normalize_module_result(
    module: MedicalModule,
    payload: dict[str, Any],
) -> NormalizedModuleResult:
    if module.module_id == "pneumonia_detection":
        return normalize_pneumonia_result(payload)

    raise UnsupportedResultNormalizationError(module)
