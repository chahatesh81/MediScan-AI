from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModuleStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    PLANNED = "PLANNED"


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"


class MedicalModality(StrEnum):
    CHEST_XRAY = "chest_xray"
    BRAIN_MRI = "brain_mri"
    DERMOSCOPY = "dermoscopy"
    BREAST_IMAGING = "breast_imaging"


@dataclass(frozen=True, slots=True)
class MedicalModule:
    module_id: str
    display_name: str
    modality: MedicalModality
    task_type: TaskType
    status: ModuleStatus
    output_classes: tuple[str, ...]
    supports_gradcam: bool
    executable: bool

    def __post_init__(self) -> None:
        if not self.module_id:
            raise ValueError(
                "module_id must not be empty."
            )

        if not self.display_name:
            raise ValueError(
                "display_name must not be empty."
            )

        if len(self.output_classes) < 2:
            raise ValueError(
                "A medical module must define at least "
                "two output classes."
            )

        if len(set(self.output_classes)) != len(
            self.output_classes
        ):
            raise ValueError(
                "output_classes must be unique."
            )

        if self.executable and (
            self.status is not ModuleStatus.AVAILABLE
        ):
            raise ValueError(
                "Only AVAILABLE modules may be executable."
            )

        if (
            self.status is ModuleStatus.PLANNED
            and self.executable
        ):
            raise ValueError(
                "PLANNED modules must not be executable."
            )


PNEUMONIA_DETECTION = MedicalModule(
    module_id="pneumonia_detection",
    display_name="Pneumonia Detection",
    modality=MedicalModality.CHEST_XRAY,
    task_type=TaskType.BINARY_CLASSIFICATION,
    status=ModuleStatus.AVAILABLE,
    output_classes=(
        "NORMAL",
        "PNEUMONIA",
    ),
    supports_gradcam=True,
    executable=True,
)


BRAIN_TUMOR_MRI = MedicalModule(
    module_id="brain_tumor_mri",
    display_name="Brain Tumor MRI Analysis",
    modality=MedicalModality.BRAIN_MRI,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    status=ModuleStatus.PLANNED,
    output_classes=(
        "GLIOMA",
        "MENINGIOMA",
        "PITUITARY_TUMOR",
        "NO_TUMOR",
    ),
    supports_gradcam=True,
    executable=False,
)


SKIN_DISEASE = MedicalModule(
    module_id="skin_disease",
    display_name="Skin Disease Analysis",
    modality=MedicalModality.DERMOSCOPY,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    status=ModuleStatus.PLANNED,
    output_classes=(
        "MELANOCYTIC_NEVUS",
        "MELANOMA",
        "BENIGN_KERATOSIS",
        "BASAL_CELL_CARCINOMA",
        "OTHER_SUPPORTED_CLASS",
    ),
    supports_gradcam=True,
    executable=False,
)


CHEST_MULTIDISEASE = MedicalModule(
    module_id="chest_multidisease",
    display_name="Multi-Disease Chest X-Ray Analysis",
    modality=MedicalModality.CHEST_XRAY,
    task_type=TaskType.MULTILABEL_CLASSIFICATION,
    status=ModuleStatus.PLANNED,
    output_classes=(
        "CARDIOMEGALY",
        "EFFUSION",
        "ATELECTASIS",
        "PNEUMONIA",
    ),
    supports_gradcam=True,
    executable=False,
)


BREAST_CANCER = MedicalModule(
    module_id="breast_cancer",
    display_name="Breast Cancer Image Analysis",
    modality=MedicalModality.BREAST_IMAGING,
    task_type=TaskType.BINARY_CLASSIFICATION,
    status=ModuleStatus.PLANNED,
    output_classes=(
        "NON_SUSPICIOUS",
        "SUSPICIOUS",
    ),
    supports_gradcam=True,
    executable=False,
)


MEDICAL_MODULES: tuple[MedicalModule, ...] = (
    PNEUMONIA_DETECTION,
    BRAIN_TUMOR_MRI,
    SKIN_DISEASE,
    CHEST_MULTIDISEASE,
    BREAST_CANCER,
)


_MODULES_BY_ID: dict[str, MedicalModule] = {
    module.module_id: module
    for module in MEDICAL_MODULES
}


if len(_MODULES_BY_ID) != len(MEDICAL_MODULES):
    raise RuntimeError(
        "Medical module IDs must be unique."
    )


def list_modules() -> tuple[MedicalModule, ...]:
    return MEDICAL_MODULES


def get_module(
    module_id: str,
) -> MedicalModule:
    try:
        return _MODULES_BY_ID[module_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown medical module: {module_id}"
        ) from exc


def list_executable_modules(
) -> tuple[MedicalModule, ...]:
    return tuple(
        module
        for module in MEDICAL_MODULES
        if module.executable
    )
