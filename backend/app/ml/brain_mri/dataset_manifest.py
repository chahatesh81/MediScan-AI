from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Iterable

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_normalization import (
    ImageRole,
    MRISequence,
    classify_member_role,
    normalize_fernando_label,
    normalize_primary_label,
    split_fernando_class,
)


class ManifestDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class BrainMRIManifestRecord:
    record_id: str
    source_id: str
    archive_name: str
    archive_member: str
    original_label: str | None
    normalized_class: BrainMRIClass | None
    sequence: MRISequence
    image_role: ImageRole
    decision: ManifestDecision
    reason: str


PRIMARY_SOURCE_IDS = frozenset(
    {
        "mendeley_12k",
        "masoud_deduplicated",
    }
)

FERNANDO_SOURCE_IDS = frozenset(
    {
        "fernando_30_class",
        "fernando_38_class",
    }
)


def stable_record_id(
    source_id: str,
    archive_member: str,
) -> str:
    payload = (
        f"{source_id}\0{archive_member}"
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def primary_label_from_member(
    archive_member: str,
) -> str | None:
    parts = PurePosixPath(
        archive_member
    ).parts

    if len(parts) < 2:
        return None

    return parts[-2]


def fernando_label_from_member(
    source_id: str,
    archive_member: str,
) -> str | None:
    parts = PurePosixPath(
        archive_member
    ).parts

    if source_id == "fernando_30_class":
        if len(parts) < 2:
            return None

        return parts[0]

    if source_id == "fernando_38_class":
        if len(parts) < 2:
            return None

        return parts[-2]

    return None


def build_image_manifest_record(
    *,
    source_id: str,
    archive_name: str,
    archive_member: str,
) -> BrainMRIManifestRecord:
    role = classify_member_role(
        archive_member
    )

    record_id = stable_record_id(
        source_id,
        archive_member,
    )

    if role is not ImageRole.SOURCE_IMAGE:
        return BrainMRIManifestRecord(
            record_id=record_id,
            source_id=source_id,
            archive_name=archive_name,
            archive_member=archive_member,
            original_label=None,
            normalized_class=None,
            sequence=MRISequence.UNKNOWN,
            image_role=role,
            decision=ManifestDecision.REJECT,
            reason="non_source_image",
        )

    if source_id in PRIMARY_SOURCE_IDS:
        original_label = (
            primary_label_from_member(
                archive_member
            )
        )

        if original_label is None:
            return BrainMRIManifestRecord(
                record_id=record_id,
                source_id=source_id,
                archive_name=archive_name,
                archive_member=archive_member,
                original_label=None,
                normalized_class=None,
                sequence=MRISequence.UNKNOWN,
                image_role=role,
                decision=ManifestDecision.REJECT,
                reason="missing_primary_label",
            )

        normalized = normalize_primary_label(
            original_label
        )

        return BrainMRIManifestRecord(
            record_id=record_id,
            source_id=source_id,
            archive_name=archive_name,
            archive_member=archive_member,
            original_label=original_label,
            normalized_class=normalized.target,
            sequence=MRISequence.UNKNOWN,
            image_role=role,
            decision=(
                ManifestDecision.ACCEPT
                if normalized.accepted
                else ManifestDecision.REJECT
            ),
            reason=normalized.reason,
        )

    if source_id in FERNANDO_SOURCE_IDS:
        original_label = (
            fernando_label_from_member(
                source_id,
                archive_member,
            )
        )

        if original_label is None:
            return BrainMRIManifestRecord(
                record_id=record_id,
                source_id=source_id,
                archive_name=archive_name,
                archive_member=archive_member,
                original_label=None,
                normalized_class=None,
                sequence=MRISequence.UNKNOWN,
                image_role=role,
                decision=ManifestDecision.REJECT,
                reason="missing_fernando_label",
            )

        _, sequence = split_fernando_class(
            original_label
        )

        normalized = normalize_fernando_label(
            original_label
        )

        return BrainMRIManifestRecord(
            record_id=record_id,
            source_id=source_id,
            archive_name=archive_name,
            archive_member=archive_member,
            original_label=original_label,
            normalized_class=normalized.target,
            sequence=sequence,
            image_role=role,
            decision=(
                ManifestDecision.ACCEPT
                if normalized.accepted
                else ManifestDecision.REJECT
            ),
            reason=normalized.reason,
        )

    return BrainMRIManifestRecord(
        record_id=record_id,
        source_id=source_id,
        archive_name=archive_name,
        archive_member=archive_member,
        original_label=None,
        normalized_class=None,
        sequence=MRISequence.UNKNOWN,
        image_role=role,
        decision=ManifestDecision.REJECT,
        reason="unsupported_manifest_source",
    )


def build_image_manifest(
    *,
    source_id: str,
    archive_name: str,
    archive_members: Iterable[str],
) -> tuple[BrainMRIManifestRecord, ...]:
    records = tuple(
        build_image_manifest_record(
            source_id=source_id,
            archive_name=archive_name,
            archive_member=member,
        )
        for member in sorted(
            archive_members
        )
    )

    record_ids = tuple(
        record.record_id
        for record in records
    )

    if len(record_ids) != len(
        set(record_ids)
    ):
        raise ValueError(
            "duplicate_manifest_record_id"
        )

    return records
