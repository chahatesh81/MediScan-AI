from __future__ import annotations

import pytest

from backend.app.ml.brain_mri.dataset_inventory import (
    BRAIN_MRI_ARCHIVE_INVENTORIES,
    BrainMRIInventoryRole,
    BrainMRISourceKind,
    archive_inventory_by_source_id,
    count_allowed_inventory_members,
    is_allowed_inventory_member,
    normalized_archive_suffix,
)
from backend.app.ml.brain_mri.dataset_source import (
    acquisition_sources,
)


def test_inventory_covers_every_acquisition_source() -> None:
    assert {
        inventory.source_id
        for inventory in BRAIN_MRI_ARCHIVE_INVENTORIES
    } == {
        source.source_id
        for source in acquisition_sources()
    }


def test_inventory_source_ids_are_unique() -> None:
    source_ids = tuple(
        inventory.source_id
        for inventory in BRAIN_MRI_ARCHIVE_INVENTORIES
    )

    assert len(source_ids) == len(set(source_ids))


def test_inventory_archive_names_are_unique() -> None:
    archive_names = tuple(
        inventory.archive_name
        for inventory in BRAIN_MRI_ARCHIVE_INVENTORIES
    )

    assert len(archive_names) == len(set(archive_names))


def test_primary_sources_are_image_classification() -> None:
    primary = tuple(
        inventory
        for inventory in BRAIN_MRI_ARCHIVE_INVENTORIES
        if inventory.inventory_role
        is BrainMRIInventoryRole.PRIMARY
    )

    assert len(primary) == 2

    assert all(
        inventory.source_kind
        is BrainMRISourceKind.IMAGE_CLASSIFICATION
        for inventory in primary
    )


def test_patient_aware_sources_are_volume_sources() -> None:
    patient_aware = tuple(
        inventory
        for inventory in BRAIN_MRI_ARCHIVE_INVENTORIES
        if inventory.inventory_role
        is BrainMRIInventoryRole.PATIENT_AWARE
    )

    assert len(patient_aware) == 3

    assert all(
        inventory.source_kind
        is BrainMRISourceKind.PATIENT_AWARE_VOLUME
        for inventory in patient_aware
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    (
        ("image.JPG", ".jpg"),
        ("scan.jpeg", ".jpeg"),
        ("case.PNG", ".png"),
        ("patient.npz", ".npz"),
        ("volume.nii", ".nii"),
        ("volume.NII.GZ", ".nii.gz"),
    ),
)
def test_normalized_archive_suffix(
    path: str,
    expected: str,
) -> None:
    assert normalized_archive_suffix(path) == expected


def test_directory_members_are_not_usable_samples() -> None:
    inventory = archive_inventory_by_source_id(
        "mendeley_12k"
    )

    assert (
        is_allowed_inventory_member(
            inventory,
            "Training/glioma/",
        )
        is False
    )


def test_primary_image_inventory_accepts_images() -> None:
    inventory = archive_inventory_by_source_id(
        "mendeley_12k"
    )

    assert is_allowed_inventory_member(
        inventory,
        "Training/glioma/example.jpg",
    )


def test_primary_image_inventory_rejects_metadata() -> None:
    inventory = archive_inventory_by_source_id(
        "mendeley_12k"
    )

    assert (
        is_allowed_inventory_member(
            inventory,
            "metadata.csv",
        )
        is False
    )


def test_brats_2021_accepts_npz_only() -> None:
    inventory = archive_inventory_by_source_id(
        "brats_2021"
    )

    assert is_allowed_inventory_member(
        inventory,
        "BraTS2021_00000.npz",
    )

    assert (
        is_allowed_inventory_member(
            inventory,
            "preview.png",
        )
        is False
    )


def test_allowed_member_count_is_deterministic() -> None:
    inventory = archive_inventory_by_source_id(
        "mendeley_12k"
    )

    members = (
        "a.jpg",
        "b.PNG",
        "metadata.csv",
        "folder/",
    )

    assert count_allowed_inventory_members(
        inventory,
        members,
    ) == 2


def test_unknown_source_id_is_rejected() -> None:
    with pytest.raises(KeyError):
        archive_inventory_by_source_id(
            "unknown_source"
        )
