from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from backend.app.modules.registry import (
    MedicalModality,
    MedicalModule,
    ModuleStatus,
    TaskType,
)


def build_module(
    **overrides: object,
) -> MedicalModule:
    values: dict[str, object] = {
        "module_id": "pneumonia_detection",
        "display_name": "Pneumonia Detection",
        "modality": MedicalModality.CHEST_XRAY,
        "task_type": TaskType.BINARY_CLASSIFICATION,
        "status": ModuleStatus.AVAILABLE,
        "output_classes": (
            "NORMAL",
            "PNEUMONIA",
        ),
        "supports_gradcam": True,
        "executable": True,
    }
    values.update(overrides)

    return MedicalModule(**values)


def test_medical_module_is_immutable() -> None:
    module = build_module()

    with pytest.raises(FrozenInstanceError):
        module.display_name = "Changed"


def test_medical_module_preserves_contract() -> None:
    module = build_module()

    assert module.module_id == "pneumonia_detection"
    assert module.status is ModuleStatus.AVAILABLE
    assert module.executable is True
    assert module.output_classes == (
        "NORMAL",
        "PNEUMONIA",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "module_id",
            "",
            "module_id must not be empty.",
        ),
        (
            "display_name",
            "",
            "display_name must not be empty.",
        ),
        (
            "output_classes",
            ("NORMAL",),
            "at least two output classes",
        ),
        (
            "output_classes",
            ("NORMAL", "NORMAL"),
            "output_classes must be unique.",
        ),
    ],
)
def test_medical_module_rejects_invalid_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        build_module(**{field: value})


def test_planned_module_cannot_be_executable() -> None:
    with pytest.raises(
        ValueError,
        match="Only AVAILABLE modules",
    ):
        build_module(
            status=ModuleStatus.PLANNED,
            executable=True,
        )


def test_planned_module_may_be_non_executable() -> None:
    module = build_module(
        status=ModuleStatus.PLANNED,
        executable=False,
    )

    assert module.status is ModuleStatus.PLANNED
    assert module.executable is False
