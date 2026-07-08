from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Iterable


class BrainMRISourceKind(StrEnum):
    IMAGE_CLASSIFICATION = "image_classification"
    CONDITIONAL_CLASSIFICATION = "conditional_classification"
    PATIENT_AWARE_VOLUME = "patient_aware_volume"


class BrainMRIInventoryRole(StrEnum):
    PRIMARY = "primary"
    CONDITIONAL = "conditional"
    PATIENT_AWARE = "patient_aware"


@dataclass(frozen=True, slots=True)
class BrainMRIArchiveInventory:
    source_id: str
    archive_name: str
    source_kind: BrainMRISourceKind
    inventory_role: BrainMRIInventoryRole
    allowed_extensions: tuple[str, ...]


BRAIN_MRI_ARCHIVE_INVENTORIES: tuple[
    BrainMRIArchiveInventory, ...
] = (
    BrainMRIArchiveInventory(
        source_id="mendeley_12k",
        archive_name="mendely-brain-tumor.zip",
        source_kind=BrainMRISourceKind.IMAGE_CLASSIFICATION,
        inventory_role=BrainMRIInventoryRole.PRIMARY,
        allowed_extensions=(
            ".jpg",
            ".jpeg",
            ".png",
        ),
    ),
    BrainMRIArchiveInventory(
        source_id="masoud_deduplicated",
        archive_name=(
            "brain-tumor-mri-deduplicated-clean-version.zip"
        ),
        source_kind=BrainMRISourceKind.IMAGE_CLASSIFICATION,
        inventory_role=BrainMRIInventoryRole.PRIMARY,
        allowed_extensions=(
            ".jpg",
            ".jpeg",
            ".png",
        ),
    ),
    BrainMRIArchiveInventory(
        source_id="fernando_30_class",
        archive_name="brain-tumor-mri-images-30-classes.zip",
        source_kind=(
            BrainMRISourceKind.CONDITIONAL_CLASSIFICATION
        ),
        inventory_role=BrainMRIInventoryRole.CONDITIONAL,
        allowed_extensions=(
            ".jpg",
            ".jpeg",
            ".png",
        ),
    ),
    BrainMRIArchiveInventory(
        source_id="fernando_38_class",
        archive_name=(
            "brain-tumor-12k-mri-images-w-masks-meta-and-bbox.zip"
        ),
        source_kind=(
            BrainMRISourceKind.CONDITIONAL_CLASSIFICATION
        ),
        inventory_role=BrainMRIInventoryRole.CONDITIONAL,
        allowed_extensions=(
            ".jpg",
            ".jpeg",
            ".png",
        ),
    ),
    BrainMRIArchiveInventory(
        source_id="brats_2021",
        archive_name="brats-2021-preprocessed-npz.zip",
        source_kind=BrainMRISourceKind.PATIENT_AWARE_VOLUME,
        inventory_role=BrainMRIInventoryRole.PATIENT_AWARE,
        allowed_extensions=(".npz",),
    ),
    BrainMRIArchiveInventory(
        source_id="brats_africa",
        archive_name="brats-africa.zip",
        source_kind=BrainMRISourceKind.PATIENT_AWARE_VOLUME,
        inventory_role=BrainMRIInventoryRole.PATIENT_AWARE,
        allowed_extensions=(
            ".nii",
            ".nii.gz",
        ),
    ),
    BrainMRIArchiveInventory(
        source_id="brats_meningioma",
        archive_name="brats-men-dataset-v1.zip",
        source_kind=BrainMRISourceKind.PATIENT_AWARE_VOLUME,
        inventory_role=BrainMRIInventoryRole.PATIENT_AWARE,
        allowed_extensions=(
            ".nii",
            ".nii.gz",
        ),
    ),
)


def archive_inventory_by_source_id(
    source_id: str,
) -> BrainMRIArchiveInventory:
    for inventory in BRAIN_MRI_ARCHIVE_INVENTORIES:
        if inventory.source_id == source_id:
            return inventory

    raise KeyError(source_id)


def normalized_archive_suffix(
    path: str,
) -> str:
    lowered = path.lower()

    if lowered.endswith(".nii.gz"):
        return ".nii.gz"

    return PurePosixPath(lowered).suffix


def is_allowed_inventory_member(
    inventory: BrainMRIArchiveInventory,
    member_path: str,
) -> bool:
    if member_path.endswith("/"):
        return False

    suffix = normalized_archive_suffix(member_path)

    return suffix in inventory.allowed_extensions


def count_allowed_inventory_members(
    inventory: BrainMRIArchiveInventory,
    member_paths: Iterable[str],
) -> int:
    return sum(
        1
        for member_path in member_paths
        if is_allowed_inventory_member(
            inventory,
            member_path,
        )
    )
