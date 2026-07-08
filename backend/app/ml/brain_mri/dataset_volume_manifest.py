from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Iterable


class BraTSVolumeFormat(StrEnum):
    NPZ = "NPZ"
    NIFTI = "NIFTI"


class BraTSModality(StrEnum):
    MULTICHANNEL = "MULTICHANNEL"
    T1 = "T1"
    T1C = "T1C"
    T2 = "T2"
    FLAIR = "FLAIR"
    SEGMENTATION = "SEGMENTATION"


@dataclass(frozen=True, slots=True)
class BraTSMember:
    source_id: str
    case_id: str
    member_path: str
    modality: BraTSModality
    volume_format: BraTSVolumeFormat


@dataclass(frozen=True, slots=True)
class BraTSCaseManifest:
    source_id: str
    case_id: str
    members: tuple[BraTSMember, ...]
    has_segmentation: bool

    @property
    def modalities(self) -> tuple[BraTSModality, ...]:
        return tuple(
            member.modality
            for member in self.members
        )


_BRATS_2021_PATTERN = re.compile(
    r"^(BraTS2021_\d{5})\.npz$"
)

_BRATS_AFRICA_CASE_PATTERN = re.compile(
    r"^(BraTS-SSA-\d{5}-\d{3})$"
)

_BRATS_MEN_CASE_PATTERN = re.compile(
    r"^(BraTS-MEN-\d{5}-\d{3})$"
)

_NIFTI_MODALITY_SUFFIXES = {
    "-t1n.nii": BraTSModality.T1,
    "-t1c.nii": BraTSModality.T1C,
    "-t2w.nii": BraTSModality.T2,
    "-t2f.nii": BraTSModality.FLAIR,
    "-seg.nii": BraTSModality.SEGMENTATION,
}


def _normalize_member_path(
    member_path: str,
) -> str:
    path = member_path.replace("\\", "/").strip("/")

    if not path:
        raise ValueError(
            "member_path must not be empty"
        )

    return path


def _parse_brats_2021_member(
    member_path: str,
) -> BraTSMember | None:
    path = PurePosixPath(member_path)

    if len(path.parts) != 1:
        return None

    match = _BRATS_2021_PATTERN.fullmatch(
        path.name
    )

    if match is None:
        return None

    return BraTSMember(
        source_id="brats_2021",
        case_id=match.group(1),
        member_path=member_path,
        modality=BraTSModality.MULTICHANNEL,
        volume_format=BraTSVolumeFormat.NPZ,
    )


def _parse_nifti_member(
    *,
    source_id: str,
    member_path: str,
    case_pattern: re.Pattern[str],
) -> BraTSMember | None:
    path = PurePosixPath(member_path)

    if len(path.parts) < 2:
        return None

    case_id = path.parent.name

    if case_pattern.fullmatch(case_id) is None:
        return None

    filename = path.name.lower()

    modality = None

    for suffix, candidate in (
        _NIFTI_MODALITY_SUFFIXES.items()
    ):
        if filename == (
            case_id.lower() + suffix
        ):
            modality = candidate
            break

    if modality is None:
        return None

    return BraTSMember(
        source_id=source_id,
        case_id=case_id,
        member_path=member_path,
        modality=modality,
        volume_format=BraTSVolumeFormat.NIFTI,
    )


def parse_brats_member(
    source_id: str,
    member_path: str,
) -> BraTSMember | None:
    normalized = _normalize_member_path(
        member_path
    )

    if source_id == "brats_2021":
        return _parse_brats_2021_member(
            normalized
        )

    if source_id == "brats_africa":
        return _parse_nifti_member(
            source_id=source_id,
            member_path=normalized,
            case_pattern=(
                _BRATS_AFRICA_CASE_PATTERN
            ),
        )

    if source_id == "brats_meningioma":
        return _parse_nifti_member(
            source_id=source_id,
            member_path=normalized,
            case_pattern=(
                _BRATS_MEN_CASE_PATTERN
            ),
        )

    raise ValueError(
        f"unsupported BraTS source: {source_id}"
    )


def _validate_case_members(
    source_id: str,
    case_id: str,
    members: tuple[BraTSMember, ...],
) -> None:
    modalities = [
        member.modality
        for member in members
    ]

    if len(modalities) != len(set(modalities)):
        raise ValueError(
            "duplicate modality for "
            f"{source_id}/{case_id}"
        )

    if source_id == "brats_2021":
        expected = {
            BraTSModality.MULTICHANNEL
        }

        if set(modalities) != expected:
            raise ValueError(
                "invalid brats_2021 case members: "
                f"{case_id}"
            )

        return

    required = {
        BraTSModality.T1,
        BraTSModality.T1C,
        BraTSModality.T2,
        BraTSModality.FLAIR,
    }

    if not required.issubset(modalities):
        missing = sorted(
            modality.value
            for modality in (
                required - set(modalities)
            )
        )

        raise ValueError(
            "missing required modalities for "
            f"{source_id}/{case_id}: {missing}"
        )

    allowed = required | {
        BraTSModality.SEGMENTATION
    }

    unexpected = set(modalities) - allowed

    if unexpected:
        raise ValueError(
            "unexpected modalities for "
            f"{source_id}/{case_id}"
        )


def build_brats_case_manifest(
    source_id: str,
    member_paths: Iterable[str],
) -> tuple[BraTSCaseManifest, ...]:
    grouped: dict[
        str,
        list[BraTSMember],
    ] = {}

    for member_path in member_paths:
        member = parse_brats_member(
            source_id,
            member_path,
        )

        if member is None:
            continue

        grouped.setdefault(
            member.case_id,
            [],
        ).append(member)

    cases: list[BraTSCaseManifest] = []

    for case_id in sorted(grouped):
        members = tuple(
            sorted(
                grouped[case_id],
                key=lambda item: (
                    item.modality.value,
                    item.member_path,
                ),
            )
        )

        _validate_case_members(
            source_id,
            case_id,
            members,
        )

        cases.append(
            BraTSCaseManifest(
                source_id=source_id,
                case_id=case_id,
                members=members,
                has_segmentation=(
                    source_id == "brats_2021"
                    or any(
                        member.modality
                        == BraTSModality.SEGMENTATION
                        for member in members
                    )
                ),
            )
        )

    return tuple(cases)
