from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Iterable


class DuplicateScope(StrEnum):
    WITHIN_SOURCE = "WITHIN_SOURCE"
    CROSS_SOURCE = "CROSS_SOURCE"


class LeakageType(StrEnum):
    CONTENT = "CONTENT"
    PATIENT = "PATIENT"


@dataclass(frozen=True, slots=True)
class DeduplicationRecord:
    record_id: str
    source_id: str
    content_sha256: str
    patient_id: str | None = None
    split: str | None = None


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    content_sha256: str
    record_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    canonical_record_id: str
    scope: DuplicateScope


@dataclass(frozen=True, slots=True)
class LeakageViolation:
    leakage_type: LeakageType
    key: str
    record_ids: tuple[str, ...]
    splits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeduplicationAudit:
    total_records: int
    unique_content_hashes: int
    duplicate_groups: tuple[DuplicateGroup, ...]
    canonical_record_ids: tuple[str, ...]
    duplicate_record_ids: tuple[str, ...]


def sha256_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def _validate_record(record: DeduplicationRecord) -> None:
    if not record.record_id:
        raise ValueError("record_id must not be empty")

    if not record.source_id:
        raise ValueError("source_id must not be empty")

    if len(record.content_sha256) != 64:
        raise ValueError(
            "content_sha256 must contain 64 hexadecimal characters"
        )

    try:
        int(record.content_sha256, 16)
    except ValueError as exc:
        raise ValueError(
            "content_sha256 must contain 64 hexadecimal characters"
        ) from exc


def _ordered_records(
    records: Iterable[DeduplicationRecord],
) -> tuple[DeduplicationRecord, ...]:
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                record.source_id,
                record.record_id,
            ),
        )
    )

    seen_record_ids: set[str] = set()

    for record in ordered:
        _validate_record(record)

        if record.record_id in seen_record_ids:
            raise ValueError(
                f"duplicate record_id: {record.record_id}"
            )

        seen_record_ids.add(record.record_id)

    return ordered


def build_duplicate_groups(
    records: Iterable[DeduplicationRecord],
) -> tuple[DuplicateGroup, ...]:
    ordered = _ordered_records(records)

    by_hash: dict[str, list[DeduplicationRecord]] = defaultdict(list)

    for record in ordered:
        by_hash[record.content_sha256].append(record)

    groups: list[DuplicateGroup] = []

    for content_hash, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue

        record_ids = tuple(
            sorted(member.record_id for member in members)
        )

        source_ids = tuple(
            sorted({member.source_id for member in members})
        )

        scope = (
            DuplicateScope.CROSS_SOURCE
            if len(source_ids) > 1
            else DuplicateScope.WITHIN_SOURCE
        )

        groups.append(
            DuplicateGroup(
                content_sha256=content_hash,
                record_ids=record_ids,
                source_ids=source_ids,
                canonical_record_id=record_ids[0],
                scope=scope,
            )
        )

    return tuple(groups)


def audit_duplicates(
    records: Iterable[DeduplicationRecord],
) -> DeduplicationAudit:
    ordered = _ordered_records(records)
    groups = build_duplicate_groups(ordered)

    duplicate_record_ids: set[str] = set()

    for group in groups:
        duplicate_record_ids.update(
            record_id
            for record_id in group.record_ids
            if record_id != group.canonical_record_id
        )

    canonical_record_ids = tuple(
        record.record_id
        for record in ordered
        if record.record_id not in duplicate_record_ids
    )

    return DeduplicationAudit(
        total_records=len(ordered),
        unique_content_hashes=len(
            {record.content_sha256 for record in ordered}
        ),
        duplicate_groups=groups,
        canonical_record_ids=canonical_record_ids,
        duplicate_record_ids=tuple(
            sorted(duplicate_record_ids)
        ),
    )


def detect_content_leakage(
    records: Iterable[DeduplicationRecord],
) -> tuple[LeakageViolation, ...]:
    ordered = _ordered_records(records)

    by_hash: dict[str, list[DeduplicationRecord]] = defaultdict(list)

    for record in ordered:
        if record.split is not None:
            by_hash[record.content_sha256].append(record)

    violations: list[LeakageViolation] = []

    for content_hash, members in sorted(by_hash.items()):
        splits = tuple(
            sorted({member.split for member in members if member.split})
        )

        if len(splits) < 2:
            continue

        violations.append(
            LeakageViolation(
                leakage_type=LeakageType.CONTENT,
                key=content_hash,
                record_ids=tuple(
                    sorted(member.record_id for member in members)
                ),
                splits=splits,
            )
        )

    return tuple(violations)


def detect_patient_leakage(
    records: Iterable[DeduplicationRecord],
) -> tuple[LeakageViolation, ...]:
    ordered = _ordered_records(records)

    by_patient: dict[str, list[DeduplicationRecord]] = defaultdict(list)

    for record in ordered:
        if record.patient_id is not None and record.split is not None:
            by_patient[record.patient_id].append(record)

    violations: list[LeakageViolation] = []

    for patient_id, members in sorted(by_patient.items()):
        splits = tuple(
            sorted({member.split for member in members if member.split})
        )

        if len(splits) < 2:
            continue

        violations.append(
            LeakageViolation(
                leakage_type=LeakageType.PATIENT,
                key=patient_id,
                record_ids=tuple(
                    sorted(member.record_id for member in members)
                ),
                splits=splits,
            )
        )

    return tuple(violations)


def assert_no_leakage(
    records: Iterable[DeduplicationRecord],
) -> None:
    ordered = _ordered_records(records)

    content_violations = detect_content_leakage(ordered)
    patient_violations = detect_patient_leakage(ordered)

    if content_violations or patient_violations:
        raise ValueError(
            "dataset leakage detected: "
            f"content={len(content_violations)}, "
            f"patient={len(patient_violations)}"
        )
