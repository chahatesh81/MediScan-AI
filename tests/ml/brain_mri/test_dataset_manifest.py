from __future__ import annotations

import pytest

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_manifest import (
    ManifestDecision,
    build_image_manifest,
    build_image_manifest_record,
    fernando_label_from_member,
    primary_label_from_member,
    stable_record_id,
)
from backend.app.ml.brain_mri.dataset_normalization import (
    ImageRole,
    MRISequence,
)


def test_record_id_is_deterministic() -> None:
    first = stable_record_id(
        "source",
        "path/image.jpg",
    )

    second = stable_record_id(
        "source",
        "path/image.jpg",
    )

    assert first == second
    assert len(first) == 64


def test_record_id_changes_with_source() -> None:
    assert stable_record_id(
        "source_a",
        "image.jpg",
    ) != stable_record_id(
        "source_b",
        "image.jpg",
    )


def test_primary_label_uses_parent_directory() -> None:
    assert primary_label_from_member(
        "Training/glioma/image.jpg"
    ) == "glioma"


def test_primary_label_requires_parent() -> None:
    assert (
        primary_label_from_member(
            "image.jpg"
        )
        is None
    )


@pytest.mark.parametrize(
    ("source_id", "member", "expected"),
    (
        (
            "fernando_30_class",
            "Glioma T1/image.jpg",
            "Glioma T1",
        ),
        (
            "fernando_38_class",
            (
                "Diffuse Gliomas/"
                "Diffuse Gliomas T1/"
                "Glioblastoma T1/"
                "image.jpg"
            ),
            "Glioblastoma T1",
        ),
    ),
)
def test_fernando_label_extraction(
    source_id: str,
    member: str,
    expected: str,
) -> None:
    assert fernando_label_from_member(
        source_id,
        member,
    ) == expected


def test_primary_glioma_is_accepted() -> None:
    record = build_image_manifest_record(
        source_id="mendeley_12k",
        archive_name="source.zip",
        archive_member=(
            "Training/glioma/image.jpg"
        ),
    )

    assert record.decision is ManifestDecision.ACCEPT
    assert record.normalized_class is BrainMRIClass.GLIOMA
    assert record.sequence is MRISequence.UNKNOWN
    assert record.image_role is ImageRole.SOURCE_IMAGE


def test_primary_pituitary_is_accepted() -> None:
    record = build_image_manifest_record(
        source_id="masoud_deduplicated",
        archive_name="source.zip",
        archive_member=(
            "Training/pituitary/image.png"
        ),
    )

    assert record.decision is ManifestDecision.ACCEPT
    assert (
        record.normalized_class
        is BrainMRIClass.PITUITARY_TUMOR
    )


def test_fernando_supported_class_is_accepted() -> None:
    record = build_image_manifest_record(
        source_id="fernando_30_class",
        archive_name="source.zip",
        archive_member=(
            "Meningioma T1C+/image.jpg"
        ),
    )

    assert record.decision is ManifestDecision.ACCEPT
    assert (
        record.normalized_class
        is BrainMRIClass.MENINGIOMA
    )
    assert record.sequence is MRISequence.T1C


def test_fernando_unsupported_class_is_rejected() -> None:
    record = build_image_manifest_record(
        source_id="fernando_30_class",
        archive_name="source.zip",
        archive_member=(
            "Schwannoma T2/image.jpg"
        ),
    )

    assert record.decision is ManifestDecision.REJECT
    assert record.normalized_class is None
    assert record.reason == "unsupported_tumor_family"


@pytest.mark.parametrize(
    ("member", "role"),
    (
        (
            "case_bbox.png",
            ImageRole.BBOX,
        ),
        (
            "case_mask_consensus.png",
            ImageRole.MASK,
        ),
        (
            "case meta.json",
            ImageRole.METADATA,
        ),
    ),
)
def test_derived_members_are_rejected(
    member: str,
    role: ImageRole,
) -> None:
    record = build_image_manifest_record(
        source_id="fernando_38_class",
        archive_name="source.zip",
        archive_member=member,
    )

    assert record.decision is ManifestDecision.REJECT
    assert record.image_role is role
    assert record.reason == "non_source_image"


def test_manifest_order_is_deterministic() -> None:
    members = (
        "Training/glioma/c.jpg",
        "Training/glioma/a.jpg",
        "Training/glioma/b.jpg",
    )

    manifest = build_image_manifest(
        source_id="mendeley_12k",
        archive_name="source.zip",
        archive_members=members,
    )

    assert tuple(
        record.archive_member
        for record in manifest
    ) == tuple(sorted(members))


def test_manifest_rejects_duplicate_record_ids() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate_manifest_record_id",
    ):
        build_image_manifest(
            source_id="mendeley_12k",
            archive_name="source.zip",
            archive_members=(
                "Training/glioma/a.jpg",
                "Training/glioma/a.jpg",
            ),
        )
