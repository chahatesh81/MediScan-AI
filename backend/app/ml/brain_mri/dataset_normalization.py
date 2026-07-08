from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


from backend.app.ml.brain_mri.contract import BrainMRIClass


class MRISequence(StrEnum):
    T1 = "T1"
    T1C = "T1C+"
    T2 = "T2"
    UNKNOWN = "UNKNOWN"


class ImageRole(StrEnum):
    SOURCE_IMAGE = "SOURCE_IMAGE"
    MASK = "MASK"
    BBOX = "BBOX"
    METADATA = "METADATA"
    VOLUME = "VOLUME"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class NormalizedLabel:
    target: BrainMRIClass | None
    accepted: bool
    reason: str


FERNANDO_SUPPORTED_TUMORS = {
    "glioma": BrainMRIClass.GLIOMA,
    "meningioma": BrainMRIClass.MENINGIOMA,
    "normal": BrainMRIClass.NO_TUMOR,
}

FERNANDO_EXCLUDED_TUMORS = frozenset(
    {
        "astrocytoma",
        "dysembryoplastic neuroepithelial tumor",
        "ependymoma",
        "ependymoma - subependymoma",
        "ganglioglioma",
        "germinoma",
        "glioblastoma",
        "hemangiopericytoma",
        "medulloblastoma",
        "neurocytoma",
        "oligodendroglioma",
        "other",
        "schwannoma",
    }
)


def classify_member_role(member: str) -> ImageRole:
    name = PurePosixPath(member).name.lower()

    if name.endswith("_mask_consensus.png"):
        return ImageRole.MASK

    if name.endswith("_bbox.png"):
        return ImageRole.BBOX

    if name.endswith(" meta.json") or name == "data.json":
        return ImageRole.METADATA

    if name.endswith((".nii", ".nii.gz", ".npz")):
        return ImageRole.VOLUME

    if name.endswith((".jpg", ".jpeg", ".png")):
        return ImageRole.SOURCE_IMAGE

    return ImageRole.OTHER


def normalize_primary_label(raw_label: str) -> NormalizedLabel:
    value = raw_label.strip().lower().replace("-", "").replace("_", "")

    mapping = {
        "glioma": BrainMRIClass.GLIOMA,
        "meningioma": BrainMRIClass.MENINGIOMA,
        "pituitary": BrainMRIClass.PITUITARY_TUMOR,
        "pituitarytumor": BrainMRIClass.PITUITARY_TUMOR,
        "notumor": BrainMRIClass.NO_TUMOR,
        "normal": BrainMRIClass.NO_TUMOR,
    }

    target = mapping.get(value)

    if target is None:
        return NormalizedLabel(
            target=None,
            accepted=False,
            reason="unsupported_primary_label",
        )

    return NormalizedLabel(
        target=target,
        accepted=True,
        reason="explicit_primary_mapping",
    )


def split_fernando_class(raw_class: str) -> tuple[str, MRISequence]:
    value = raw_class.strip()

    sequence_patterns = (
        (" T1C+", MRISequence.T1C),
        (" T1", MRISequence.T1),
        (" T2", MRISequence.T2),
    )

    for suffix, sequence in sequence_patterns:
        if value.upper().endswith(suffix.upper()):
            tumor = value[: -len(suffix)].strip()
            return tumor, sequence

    return value, MRISequence.UNKNOWN


def normalize_fernando_label(raw_class: str) -> NormalizedLabel:
    tumor, _ = split_fernando_class(raw_class)
    normalized = tumor.strip().lower()

    target = FERNANDO_SUPPORTED_TUMORS.get(normalized)

    if target is not None:
        return NormalizedLabel(
            target=target,
            accepted=True,
            reason="explicit_fernando_mapping",
        )

    if normalized in FERNANDO_EXCLUDED_TUMORS:
        return NormalizedLabel(
            target=None,
            accepted=False,
            reason="unsupported_tumor_family",
        )

    return NormalizedLabel(
        target=None,
        accepted=False,
        reason="unknown_fernando_label",
    )


def brats_case_id(member: str) -> str:
    name = PurePosixPath(member).name

    for suffix in (".nii.gz", ".nii", ".npz"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break

    return re.sub(
        r"(?i)-(seg|t1c|t1n|t2f|t2w)$",
        "",
        name,
    )


def brats_modality(member: str) -> str | None:
    name = PurePosixPath(member).name.lower()

    for modality in ("t1c", "t1n", "t2f", "t2w", "seg"):
        if re.search(rf"-{modality}(?:\.nii(?:\.gz)?)$", name):
            return modality

    return None
