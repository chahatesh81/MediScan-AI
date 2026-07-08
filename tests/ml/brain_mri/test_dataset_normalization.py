import pytest

from backend.app.ml.brain_mri.dataset_normalization import (
    BrainMRIClass,
    ImageRole,
    MRISequence,
    brats_case_id,
    brats_modality,
    classify_member_role,
    normalize_fernando_label,
    normalize_primary_label,
    split_fernando_class,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("glioma", BrainMRIClass.GLIOMA),
        ("meningioma", BrainMRIClass.MENINGIOMA),
        ("pituitary", BrainMRIClass.PITUITARY_TUMOR),
        ("pituitary_tumor", BrainMRIClass.PITUITARY_TUMOR),
        ("notumor", BrainMRIClass.NO_TUMOR),
        ("normal", BrainMRIClass.NO_TUMOR),
    ],
)
def test_primary_labels_are_explicit(raw, expected):
    result = normalize_primary_label(raw)

    assert result.accepted is True
    assert result.target is expected


def test_unknown_primary_label_is_rejected():
    result = normalize_primary_label("other")

    assert result.accepted is False
    assert result.target is None


@pytest.mark.parametrize(
    ("raw", "tumor", "sequence"),
    [
        ("Glioma T1", "Glioma", MRISequence.T1),
        ("Glioma T1C+", "Glioma", MRISequence.T1C),
        ("Glioma T2", "Glioma", MRISequence.T2),
        ("Normal T1C+", "Normal", MRISequence.T1C),
    ],
)
def test_fernando_class_parsing(raw, tumor, sequence):
    assert split_fernando_class(raw) == (tumor, sequence)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Glioma T1", BrainMRIClass.GLIOMA),
        ("Meningioma T2", BrainMRIClass.MENINGIOMA),
        ("Normal T1C+", BrainMRIClass.NO_TUMOR),
    ],
)
def test_supported_fernando_labels(raw, expected):
    result = normalize_fernando_label(raw)

    assert result.accepted is True
    assert result.target is expected


@pytest.mark.parametrize(
    "raw",
    [
        "Astrocytoma T1",
        "Glioblastoma T1C+",
        "Medulloblastoma T2",
        "Schwannoma T1",
        "Other T2",
    ],
)
def test_unsupported_tumor_families_are_rejected(raw):
    result = normalize_fernando_label(raw)

    assert result.accepted is False
    assert result.target is None
    assert result.reason == "unsupported_tumor_family"


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("case.jpg", ImageRole.SOURCE_IMAGE),
        ("case.png", ImageRole.SOURCE_IMAGE),
        ("case_bbox.png", ImageRole.BBOX),
        ("case_mask_consensus.png", ImageRole.MASK),
        ("case meta.json", ImageRole.METADATA),
        ("DATA.json", ImageRole.METADATA),
        ("case.nii", ImageRole.VOLUME),
        ("case.nii.gz", ImageRole.VOLUME),
        ("case.npz", ImageRole.VOLUME),
    ],
)
def test_member_roles(member, expected):
    assert classify_member_role(member) is expected


@pytest.mark.parametrize(
    "member",
    [
        "BraTS-SSA-00009-000-seg.nii",
        "BraTS-SSA-00009-000-t1c.nii",
        "BraTS-SSA-00009-000-t1n.nii",
        "BraTS-SSA-00009-000-t2f.nii",
        "BraTS-SSA-00009-000-t2w.nii",
    ],
)
def test_brats_modalities_share_case_identity(member):
    assert brats_case_id(member) == "BraTS-SSA-00009-000"


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("case-t1c.nii", "t1c"),
        ("case-t1n.nii", "t1n"),
        ("case-t2f.nii", "t2f"),
        ("case-t2w.nii", "t2w"),
        ("case-seg.nii", "seg"),
    ],
)
def test_brats_modality(member, expected):
    assert brats_modality(member) == expected


def test_npz_case_identity_is_preserved():
    assert (
        brats_case_id("BraTS2021_00000.npz")
        == "BraTS2021_00000"
    )
