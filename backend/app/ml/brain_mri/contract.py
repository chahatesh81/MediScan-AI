from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BrainMRIClass(StrEnum):
    GLIOMA = "GLIOMA"
    MENINGIOMA = "MENINGIOMA"
    PITUITARY_TUMOR = "PITUITARY_TUMOR"
    NO_TUMOR = "NO_TUMOR"


BRAIN_MRI_CLASSES: tuple[BrainMRIClass, ...] = (
    BrainMRIClass.GLIOMA,
    BrainMRIClass.MENINGIOMA,
    BrainMRIClass.PITUITARY_TUMOR,
    BrainMRIClass.NO_TUMOR,
)


@dataclass(frozen=True, slots=True)
class BrainMRITechnicalContract:
    module_id: str
    classes: tuple[BrainMRIClass, ...]
    task_type: str
    split_unit: str
    patient_overlap_allowed: bool
    duplicate_overlap_allowed: bool
    test_split_locked_before_training: bool


BRAIN_MRI_TECHNICAL_CONTRACT = BrainMRITechnicalContract(
    module_id="brain_tumor_mri",
    classes=BRAIN_MRI_CLASSES,
    task_type="multiclass_classification",
    split_unit="patient_when_identifiers_available",
    patient_overlap_allowed=False,
    duplicate_overlap_allowed=False,
    test_split_locked_before_training=True,
)
