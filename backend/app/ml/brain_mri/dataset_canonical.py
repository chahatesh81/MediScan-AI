from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from backend.app.ml.brain_mri.contract import BrainMRIClass


SOURCE_PRIORITY: tuple[str, ...] = (
    "mendeley_12k",
    "masoud_deduplicated",
    "fernando_30_class",
    "fernando_38_class",
)

_SOURCE_RANK = {
    source_id: rank
    for rank, source_id in enumerate(SOURCE_PRIORITY)
}


@dataclass(frozen=True, slots=True)
class CanonicalCandidate:
    record_id: str
    source_id: str
    archive_name: str
    archive_member: str
    normalized_class: BrainMRIClass
    content_sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    record_id: str
    source_id: str
    archive_name: str
    archive_member: str
    normalized_class: BrainMRIClass
    content_sha256: str
    duplicate_group_size: int


@dataclass(frozen=True, slots=True)
class CanonicalDatasetAudit:
    input_records: int
    unique_content_hashes: int
    duplicate_records_removed: int
    canonical_records: tuple[CanonicalRecord, ...]
    class_counts: tuple[tuple[BrainMRIClass, int], ...]
    source_counts: tuple[tuple[str, int], ...]


def _validate_sha256(value: str) -> None:
    if len(value) != 64:
        raise ValueError(
            "content_sha256 must contain exactly 64 hexadecimal characters"
        )

    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            "content_sha256 must contain only hexadecimal characters"
        ) from exc


def _validate_candidate(candidate: CanonicalCandidate) -> None:
    if not candidate.record_id.strip():
        raise ValueError("record_id must not be empty")

    if not candidate.source_id.strip():
        raise ValueError("source_id must not be empty")

    if not candidate.archive_name.strip():
        raise ValueError("archive_name must not be empty")

    if not candidate.archive_member.strip():
        raise ValueError("archive_member must not be empty")

    _validate_sha256(candidate.content_sha256)


def canonical_priority_key(
    candidate: CanonicalCandidate,
) -> tuple[int, str, str, str]:
    return (
        _SOURCE_RANK.get(
            candidate.source_id,
            len(SOURCE_PRIORITY),
        ),
        candidate.source_id,
        candidate.archive_member,
        candidate.record_id,
    )


def select_canonical_candidate(
    candidates: Iterable[CanonicalCandidate],
) -> CanonicalCandidate:
    candidates_tuple = tuple(candidates)

    if not candidates_tuple:
        raise ValueError(
            "cannot select a canonical record from an empty group"
        )

    for candidate in candidates_tuple:
        _validate_candidate(candidate)

    content_hashes = {
        candidate.content_sha256
        for candidate in candidates_tuple
    }

    if len(content_hashes) != 1:
        raise ValueError(
            "all candidates in a duplicate group must share one content hash"
        )

    classes = {
        candidate.normalized_class
        for candidate in candidates_tuple
    }

    if len(classes) != 1:
        raise ValueError(
            "duplicate group contains conflicting normalized classes"
        )

    return min(
        candidates_tuple,
        key=canonical_priority_key,
    )


def build_canonical_dataset(
    candidates: Iterable[CanonicalCandidate],
) -> CanonicalDatasetAudit:
    candidates_tuple = tuple(candidates)

    record_ids = [
        candidate.record_id
        for candidate in candidates_tuple
    ]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "canonical dataset input contains duplicate record IDs"
        )

    groups: dict[str, list[CanonicalCandidate]] = defaultdict(list)

    for candidate in candidates_tuple:
        _validate_candidate(candidate)
        groups[candidate.content_sha256].append(candidate)

    canonical_records: list[CanonicalRecord] = []

    for content_sha256 in sorted(groups):
        group = tuple(groups[content_sha256])
        selected = select_canonical_candidate(group)

        canonical_records.append(
            CanonicalRecord(
                record_id=selected.record_id,
                source_id=selected.source_id,
                archive_name=selected.archive_name,
                archive_member=selected.archive_member,
                normalized_class=selected.normalized_class,
                content_sha256=selected.content_sha256,
                duplicate_group_size=len(group),
            )
        )

    canonical_records.sort(
        key=lambda record: (
            record.normalized_class.value,
            record.source_id,
            record.archive_member,
            record.record_id,
        )
    )

    class_counter = Counter(
        record.normalized_class
        for record in canonical_records
    )

    source_counter = Counter(
        record.source_id
        for record in canonical_records
    )

    return CanonicalDatasetAudit(
        input_records=len(candidates_tuple),
        unique_content_hashes=len(groups),
        duplicate_records_removed=(
            len(candidates_tuple) - len(groups)
        ),
        canonical_records=tuple(canonical_records),
        class_counts=tuple(
            sorted(
                class_counter.items(),
                key=lambda item: item[0].value,
            )
        ),
        source_counts=tuple(
            sorted(source_counter.items())
        ),
    )


def assert_canonical_dataset_integrity(
    audit: CanonicalDatasetAudit,
) -> None:
    records = audit.canonical_records

    if len(records) != audit.unique_content_hashes:
        raise ValueError(
            "canonical record count does not match unique content hash count"
        )

    if (
        audit.input_records
        - audit.duplicate_records_removed
        != audit.unique_content_hashes
    ):
        raise ValueError(
            "canonical dataset accounting is inconsistent"
        )

    record_ids = [
        record.record_id
        for record in records
    ]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "canonical dataset contains duplicate record IDs"
        )

    content_hashes = [
        record.content_sha256
        for record in records
    ]

    if len(content_hashes) != len(set(content_hashes)):
        raise ValueError(
            "canonical dataset contains duplicate content hashes"
        )

    if any(
        record.duplicate_group_size < 1
        for record in records
    ):
        raise ValueError(
            "duplicate_group_size must be at least one"
        )
