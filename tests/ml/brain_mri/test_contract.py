from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from backend.app.ml.brain_mri.contract import (
    BRAIN_MRI_CLASSES,
    BRAIN_MRI_TECHNICAL_CONTRACT,
    BrainMRIClass,
)
from backend.app.modules.registry import BRAIN_TUMOR_MRI


def test_brain_mri_classes_are_exact_and_ordered() -> None:
    assert BRAIN_MRI_CLASSES == (
        BrainMRIClass.GLIOMA,
        BrainMRIClass.MENINGIOMA,
        BrainMRIClass.PITUITARY_TUMOR,
        BrainMRIClass.NO_TUMOR,
    )


def test_brain_mri_contract_matches_module_registry() -> None:
    assert BRAIN_MRI_TECHNICAL_CONTRACT.module_id == (
        BRAIN_TUMOR_MRI.module_id
    )

    assert tuple(
        label.value
        for label in BRAIN_MRI_TECHNICAL_CONTRACT.classes
    ) == BRAIN_TUMOR_MRI.output_classes

    assert BRAIN_MRI_TECHNICAL_CONTRACT.task_type == (
        BRAIN_TUMOR_MRI.task_type.value
    )


def test_brain_mri_contract_forbids_patient_overlap() -> None:
    assert (
        BRAIN_MRI_TECHNICAL_CONTRACT.patient_overlap_allowed
        is False
    )


def test_brain_mri_contract_forbids_duplicate_overlap() -> None:
    assert (
        BRAIN_MRI_TECHNICAL_CONTRACT.duplicate_overlap_allowed
        is False
    )


def test_brain_mri_test_split_is_locked_before_training() -> None:
    assert (
        BRAIN_MRI_TECHNICAL_CONTRACT
        .test_split_locked_before_training
        is True
    )


def test_brain_mri_contract_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        BRAIN_MRI_TECHNICAL_CONTRACT.module_id = "changed"
