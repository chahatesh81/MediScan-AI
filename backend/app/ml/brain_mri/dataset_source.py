from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BrainMRISourceRole(StrEnum):
    PRIMARY = "primary"
    CONDITIONAL = "conditional"
    PATIENT_AWARE = "patient_aware"
    EXTERNAL_ONLY = "external_only"
    REJECTED_OVERLAP = "rejected_overlap"


@dataclass(frozen=True, slots=True)
class BrainMRIDatasetSource:
    source_id: str
    kaggle_ref: str
    role: BrainMRISourceRole
    expected_raw_samples: int | None
    patient_safe_split_required: bool
    source_family: str
    reason: str


BRAIN_MRI_DATASET_SOURCES: tuple[
    BrainMRIDatasetSource,
    ...,
] = (
    BrainMRIDatasetSource(
        source_id="mendeley_12k",
        kaggle_ref=(
            "arvinthcinmayangk/"
            "mendely-brain-tumor"
        ),
        role=BrainMRISourceRole.PRIMARY,
        expected_raw_samples=12064,
        patient_safe_split_required=False,
        source_family="mendeley_zwr4ntf94j",
        reason=(
            "Direct four-class MRI classification source."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="masoud_deduplicated",
        kaggle_ref=(
            "mylee77/"
            "brain-tumor-mri-deduplicated-clean-version"
        ),
        role=BrainMRISourceRole.PRIMARY,
        expected_raw_samples=6507,
        patient_safe_split_required=False,
        source_family="masoud_figshare_sartaj_br35h",
        reason=(
            "Deduplicated replacement for the original "
            "Masoud four-class dataset."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="fernando_30_class",
        kaggle_ref=(
            "fernando2rad/"
            "brain-tumor-mri-images-30-classes"
        ),
        role=BrainMRISourceRole.CONDITIONAL,
        expected_raw_samples=11300,
        patient_safe_split_required=True,
        source_family="fernando_nexus_30_class",
        reason=(
            "Only compatible glioma, meningioma, and "
            "normal categories may be remapped."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="fernando_38_class",
        kaggle_ref=(
            "fernando2rad/"
            "brain-tumor-12k-mri-images-w-masks-meta-and-bbox"
        ),
        role=BrainMRISourceRole.CONDITIONAL,
        expected_raw_samples=12643,
        patient_safe_split_required=True,
        source_family="fernando_nexus_38_class",
        reason=(
            "Only original MRI images from compatible "
            "diagnostic categories may be used; masks "
            "and bounding boxes are not samples."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="brats_2021",
        kaggle_ref=(
            "rayanalhaiek/"
            "brats-2021-preprocessed-npz"
        ),
        role=BrainMRISourceRole.PATIENT_AWARE,
        expected_raw_samples=None,
        patient_safe_split_required=True,
        source_family="brats_2021",
        reason=(
            "Volumetric glioma source requiring "
            "patient-level splitting before slicing."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="brats_africa",
        kaggle_ref="yakunokamizen/brats-africa",
        role=BrainMRISourceRole.PATIENT_AWARE,
        expected_raw_samples=146,
        patient_safe_split_required=True,
        source_family="brats_africa",
        reason=(
            "Independent volumetric glioma population "
            "requiring patient-level splitting."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="brats_meningioma",
        kaggle_ref="dukhailly/brats-men-dataset-v1",
        role=BrainMRISourceRole.PATIENT_AWARE,
        expected_raw_samples=None,
        patient_safe_split_required=True,
        source_family="brats_2023_men",
        reason=(
            "Volumetric meningioma source requiring "
            "patient-level splitting before slicing."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="mcnd",
        kaggle_ref=(
            "alifatahi/"
            "multi-class-neurological-disorder-mcnd-dataset"
        ),
        role=BrainMRISourceRole.REJECTED_OVERLAP,
        expected_raw_samples=16400,
        patient_safe_split_required=False,
        source_family="masoud_figshare_sartaj_br35h",
        reason=(
            "Brain-tumor classes derive from an already "
            "selected Masoud source family."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="consolidated_masoud_brisc",
        kaggle_ref=(
            "ramanarasimhakanduri/"
            "brain-tumour-dataset"
        ),
        role=BrainMRISourceRole.REJECTED_OVERLAP,
        expected_raw_samples=6892,
        patient_safe_split_required=False,
        source_family="masoud_brisc_consolidated",
        reason=(
            "Consolidated derivative overlaps existing "
            "source families and is not independent."
        ),
    ),
    BrainMRIDatasetSource(
        source_id="brats_pediatric",
        kaggle_ref=(
            "awansaad6797/"
            "3d-brain-tumor-mri-slices"
        ),
        role=BrainMRISourceRole.EXTERNAL_ONLY,
        expected_raw_samples=None,
        patient_safe_split_required=True,
        source_family="brats_pediatric",
        reason=(
            "Pediatric tumor distribution is reserved "
            "for external robustness evaluation."
        ),
    ),
)


def source_by_id(
    source_id: str,
) -> BrainMRIDatasetSource:
    for source in BRAIN_MRI_DATASET_SOURCES:
        if source.source_id == source_id:
            return source

    raise KeyError(source_id)


def acquisition_sources() -> tuple[
    BrainMRIDatasetSource,
    ...,
]:
    allowed_roles = {
        BrainMRISourceRole.PRIMARY,
        BrainMRISourceRole.CONDITIONAL,
        BrainMRISourceRole.PATIENT_AWARE,
    }

    return tuple(
        source
        for source in BRAIN_MRI_DATASET_SOURCES
        if source.role in allowed_roles
    )
