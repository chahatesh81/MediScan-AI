from __future__ import annotations

import pytest

from backend.app.modules.registry import (
    BRAIN_TUMOR_MRI,
    BREAST_CANCER,
    CHEST_MULTIDISEASE,
    MEDICAL_MODULES,
    PNEUMONIA_DETECTION,
    SKIN_DISEASE,
    MedicalModality,
    ModuleStatus,
    TaskType,
    get_module,
    list_executable_modules,
    list_modules,
)


def test_registry_contains_exactly_five_modules() -> None:
    assert len(MEDICAL_MODULES) == 5


def test_registry_module_ids_are_exact() -> None:
    assert tuple(
        module.module_id
        for module in MEDICAL_MODULES
    ) == (
        "pneumonia_detection",
        "brain_tumor_mri",
        "skin_disease",
        "chest_multidisease",
        "breast_cancer",
    )


def test_only_pneumonia_is_available() -> None:
    available = tuple(
        module.module_id
        for module in MEDICAL_MODULES
        if module.status is ModuleStatus.AVAILABLE
    )

    assert available == (
        "pneumonia_detection",
    )


def test_only_pneumonia_is_executable() -> None:
    executable = list_executable_modules()

    assert executable == (
        PNEUMONIA_DETECTION,
    )


def test_pneumonia_contract_matches_production() -> None:
    assert (
        PNEUMONIA_DETECTION.modality
        is MedicalModality.CHEST_XRAY
    )
    assert (
        PNEUMONIA_DETECTION.task_type
        is TaskType.BINARY_CLASSIFICATION
    )
    assert PNEUMONIA_DETECTION.output_classes == (
        "NORMAL",
        "PNEUMONIA",
    )
    assert PNEUMONIA_DETECTION.supports_gradcam is True


def test_brain_tumor_contract() -> None:
    assert (
        BRAIN_TUMOR_MRI.task_type
        is TaskType.MULTICLASS_CLASSIFICATION
    )
    assert BRAIN_TUMOR_MRI.output_classes == (
        "GLIOMA",
        "MENINGIOMA",
        "PITUITARY_TUMOR",
        "NO_TUMOR",
    )
    assert BRAIN_TUMOR_MRI.executable is False


def test_skin_disease_is_planned_only() -> None:
    assert SKIN_DISEASE.status is ModuleStatus.PLANNED
    assert SKIN_DISEASE.executable is False


def test_chest_multidisease_uses_multilabel_task() -> None:
    assert (
        CHEST_MULTIDISEASE.task_type
        is TaskType.MULTILABEL_CLASSIFICATION
    )
    assert CHEST_MULTIDISEASE.executable is False


def test_breast_cancer_remains_modality_neutral() -> None:
    assert (
        BREAST_CANCER.modality
        is MedicalModality.BREAST_IMAGING
    )
    assert BREAST_CANCER.status is ModuleStatus.PLANNED
    assert BREAST_CANCER.executable is False


@pytest.mark.parametrize(
    "module",
    MEDICAL_MODULES,
)
def test_get_module_returns_registered_identity(
    module,
) -> None:
    assert get_module(module.module_id) is module


def test_get_module_rejects_unknown_id() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown medical module",
    ):
        get_module("unknown_module")


def test_list_modules_preserves_registry_order() -> None:
    assert list_modules() is MEDICAL_MODULES


def test_all_planned_modules_are_non_executable() -> None:
    planned = tuple(
        module
        for module in MEDICAL_MODULES
        if module.status is ModuleStatus.PLANNED
    )

    assert len(planned) == 4
    assert all(
        module.executable is False
        for module in planned
    )
