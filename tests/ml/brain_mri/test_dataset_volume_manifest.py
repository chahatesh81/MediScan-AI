from __future__ import annotations

import pytest

from backend.app.ml.brain_mri.dataset_volume_manifest import (
    BraTSModality,
    BraTSVolumeFormat,
    build_brats_case_manifest,
    parse_brats_member,
)


def test_brats_2021_npz_case_identity() -> None:
    member = parse_brats_member(
        "brats_2021",
        "BraTS2021_00042.npz",
    )

    assert member is not None
    assert member.case_id == "BraTS2021_00042"
    assert (
        member.modality
        == BraTSModality.MULTICHANNEL
    )
    assert (
        member.volume_format
        == BraTSVolumeFormat.NPZ
    )


def test_brats_2021_metadata_is_ignored() -> None:
    assert (
        parse_brats_member(
            "brats_2021",
            "dataset_splits.json",
        )
        is None
    )


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("t1n", BraTSModality.T1),
        ("t1c", BraTSModality.T1C),
        ("t2w", BraTSModality.T2),
        ("t2f", BraTSModality.FLAIR),
        (
            "seg",
            BraTSModality.SEGMENTATION,
        ),
    ],
)
def test_brats_africa_modalities(
    suffix: str,
    expected: BraTSModality,
) -> None:
    case_id = "BraTS-SSA-00009-000"

    path = (
        "PKG - BraTS-Africa/"
        "BraTS-Africa/"
        "51_OtherNeoplasms/"
        f"{case_id}/"
        f"{case_id}-{suffix}.nii"
    )

    member = parse_brats_member(
        "brats_africa",
        path,
    )

    assert member is not None
    assert member.case_id == case_id
    assert member.modality == expected


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("t1n", BraTSModality.T1),
        ("t1c", BraTSModality.T1C),
        ("t2w", BraTSModality.T2),
        ("t2f", BraTSModality.FLAIR),
        (
            "seg",
            BraTSModality.SEGMENTATION,
        ),
    ],
)
def test_brats_meningioma_modalities(
    suffix: str,
    expected: BraTSModality,
) -> None:
    case_id = "BraTS-MEN-00004-000"

    path = (
        "root/train/"
        f"{case_id}/"
        f"{case_id}-{suffix}.nii"
    )

    member = parse_brats_member(
        "brats_meningioma",
        path,
    )

    assert member is not None
    assert member.case_id == case_id
    assert member.modality == expected


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported BraTS source",
    ):
        parse_brats_member(
            "unknown",
            "case.nii",
        )


def test_case_manifest_is_deterministic() -> None:
    case_id = "BraTS-SSA-00009-000"

    members = [
        f"root/{case_id}/{case_id}-t2w.nii",
        f"root/{case_id}/{case_id}-seg.nii",
        f"root/{case_id}/{case_id}-t1c.nii",
        f"root/{case_id}/{case_id}-t2f.nii",
        f"root/{case_id}/{case_id}-t1n.nii",
    ]

    forward = build_brats_case_manifest(
        "brats_africa",
        members,
    )

    reverse = build_brats_case_manifest(
        "brats_africa",
        reversed(members),
    )

    assert forward == reverse


def test_training_case_has_segmentation() -> None:
    case_id = "BraTS-MEN-00004-000"

    members = [
        f"root/{case_id}/{case_id}-t1n.nii",
        f"root/{case_id}/{case_id}-t1c.nii",
        f"root/{case_id}/{case_id}-t2w.nii",
        f"root/{case_id}/{case_id}-t2f.nii",
        f"root/{case_id}/{case_id}-seg.nii",
    ]

    manifest = build_brats_case_manifest(
        "brats_meningioma",
        members,
    )

    assert len(manifest) == 1
    assert manifest[0].has_segmentation is True


def test_validation_case_without_segmentation() -> None:
    case_id = "BraTS-MEN-01374-000"

    members = [
        f"root/{case_id}/{case_id}-t1n.nii",
        f"root/{case_id}/{case_id}-t1c.nii",
        f"root/{case_id}/{case_id}-t2w.nii",
        f"root/{case_id}/{case_id}-t2f.nii",
    ]

    manifest = build_brats_case_manifest(
        "brats_meningioma",
        members,
    )

    assert len(manifest) == 1
    assert manifest[0].has_segmentation is False


def test_missing_required_modality_is_rejected() -> None:
    case_id = "BraTS-SSA-00009-000"

    members = [
        f"root/{case_id}/{case_id}-t1n.nii",
        f"root/{case_id}/{case_id}-t1c.nii",
        f"root/{case_id}/{case_id}-t2w.nii",
    ]

    with pytest.raises(
        ValueError,
        match="missing required modalities",
    ):
        build_brats_case_manifest(
            "brats_africa",
            members,
        )


def test_duplicate_modality_is_rejected() -> None:
    case_id = "BraTS-SSA-00009-000"

    members = [
        f"a/{case_id}/{case_id}-t1n.nii",
        f"b/{case_id}/{case_id}-t1n.nii",
        f"a/{case_id}/{case_id}-t1c.nii",
        f"a/{case_id}/{case_id}-t2w.nii",
        f"a/{case_id}/{case_id}-t2f.nii",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate modality",
    ):
        build_brats_case_manifest(
            "brats_africa",
            members,
        )


def test_multiple_cases_are_sorted() -> None:
    paths = [
        "BraTS2021_00042.npz",
        "BraTS2021_00002.npz",
    ]

    manifest = build_brats_case_manifest(
        "brats_2021",
        paths,
    )

    assert [
        case.case_id
        for case in manifest
    ] == [
        "BraTS2021_00002",
        "BraTS2021_00042",
    ]


def test_brats_2021_case_has_embedded_segmentation() -> None:
    manifest = build_brats_case_manifest(
        "brats_2021",
        ["BraTS2021_00042.npz"],
    )

    assert len(manifest) == 1
    assert manifest[0].has_segmentation is True
