from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import floor
from typing import Iterable

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_canonical import CanonicalRecord


DEFAULT_SPLIT_SEED = 42


class DatasetSplit(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    record_id: str
    content_sha256: str
    normalized_class: BrainMRIClass
    split: DatasetSplit


@dataclass(frozen=True, slots=True)
class DatasetSplitAudit:
    total_records: int
    seed: int
    ratios: SplitRatios
    assignments: tuple[SplitAssignment, ...]
    split_counts: tuple[tuple[DatasetSplit, int], ...]
    class_split_counts: tuple[
        tuple[BrainMRIClass, DatasetSplit, int],
        ...,
    ]


def validate_split_ratios(ratios: SplitRatios) -> None:
    values = (
        ratios.train,
        ratios.validation,
        ratios.test,
    )

    if any(value <= 0.0 for value in values):
        raise ValueError(
            "all split ratios must be greater than zero"
        )

    if any(value >= 1.0 for value in values):
        raise ValueError(
            "each split ratio must be less than one"
        )

    if abs(sum(values) - 1.0) > 1e-12:
        raise ValueError(
            "split ratios must sum to exactly one"
        )


def _stable_split_key(
    record: CanonicalRecord,
    *,
    seed: int,
) -> tuple[str, str]:
    payload = (
        f"{seed}\0"
        f"{record.normalized_class.value}\0"
        f"{record.content_sha256}\0"
        f"{record.record_id}"
    ).encode("utf-8")

    return (
        sha256(payload).hexdigest(),
        record.record_id,
    )


def _allocate_class_counts(
    total: int,
    ratios: SplitRatios,
) -> tuple[int, int, int]:
    if total < 0:
        raise ValueError("class record count must not be negative")

    if total == 0:
        return (0, 0, 0)

    raw = (
        total * ratios.train,
        total * ratios.validation,
        total * ratios.test,
    )

    counts = [
        floor(value)
        for value in raw
    ]

    remaining = total - sum(counts)

    remainders = [
        raw[index] - counts[index]
        for index in range(3)
    ]

    priority = sorted(
        range(3),
        key=lambda index: (
            -remainders[index],
            index,
        ),
    )

    for index in priority[:remaining]:
        counts[index] += 1

    return (
        counts[0],
        counts[1],
        counts[2],
    )


def build_stratified_split(
    records: Iterable[CanonicalRecord],
    *,
    ratios: SplitRatios = SplitRatios(),
    seed: int = DEFAULT_SPLIT_SEED,
) -> DatasetSplitAudit:
    validate_split_ratios(ratios)

    records_tuple = tuple(records)

    record_ids = [
        record.record_id
        for record in records_tuple
    ]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "split input contains duplicate record IDs"
        )

    content_hashes = [
        record.content_sha256
        for record in records_tuple
    ]

    if len(content_hashes) != len(set(content_hashes)):
        raise ValueError(
            "split input contains duplicate content hashes"
        )

    records_by_class: dict[
        BrainMRIClass,
        list[CanonicalRecord],
    ] = {
        brain_class: []
        for brain_class in BrainMRIClass
    }

    for record in records_tuple:
        records_by_class[record.normalized_class].append(record)

    assignments: list[SplitAssignment] = []

    for brain_class in BrainMRIClass:
        class_records = sorted(
            records_by_class[brain_class],
            key=lambda record: _stable_split_key(
                record,
                seed=seed,
            ),
        )

        train_count, validation_count, test_count = (
            _allocate_class_counts(
                len(class_records),
                ratios,
            )
        )

        train_end = train_count
        validation_end = train_end + validation_count

        for index, record in enumerate(class_records):
            if index < train_end:
                split = DatasetSplit.TRAIN
            elif index < validation_end:
                split = DatasetSplit.VALIDATION
            else:
                split = DatasetSplit.TEST

            assignments.append(
                SplitAssignment(
                    record_id=record.record_id,
                    content_sha256=record.content_sha256,
                    normalized_class=record.normalized_class,
                    split=split,
                )
            )

        if (
            train_count
            + validation_count
            + test_count
            != len(class_records)
        ):
            raise ValueError(
                "class split allocation is inconsistent"
            )

    assignments.sort(
        key=lambda assignment: (
            assignment.split.value,
            assignment.normalized_class.value,
            assignment.record_id,
        )
    )

    split_counter = Counter(
        assignment.split
        for assignment in assignments
    )

    class_split_counter = Counter(
        (
            assignment.normalized_class,
            assignment.split,
        )
        for assignment in assignments
    )

    return DatasetSplitAudit(
        total_records=len(records_tuple),
        seed=seed,
        ratios=ratios,
        assignments=tuple(assignments),
        split_counts=tuple(
            (
                split,
                split_counter.get(split, 0),
            )
            for split in DatasetSplit
        ),
        class_split_counts=tuple(
            (
                brain_class,
                split,
                class_split_counter.get(
                    (brain_class, split),
                    0,
                ),
            )
            for brain_class in BrainMRIClass
            for split in DatasetSplit
        ),
    )


def assert_split_integrity(
    audit: DatasetSplitAudit,
) -> None:
    validate_split_ratios(audit.ratios)

    assignments = audit.assignments

    if len(assignments) != audit.total_records:
        raise ValueError(
            "assignment count does not match total record count"
        )

    record_ids = [
        assignment.record_id
        for assignment in assignments
    ]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "records are assigned to more than one split"
        )

    content_hashes = [
        assignment.content_sha256
        for assignment in assignments
    ]

    if len(content_hashes) != len(set(content_hashes)):
        raise ValueError(
            "content hashes occur in more than one assignment"
        )

    split_counter = Counter(
        assignment.split
        for assignment in assignments
    )

    if dict(audit.split_counts) != {
        split: split_counter.get(split, 0)
        for split in DatasetSplit
    }:
        raise ValueError(
            "reported split counts are inconsistent"
        )

    class_split_counter = Counter(
        (
            assignment.normalized_class,
            assignment.split,
        )
        for assignment in assignments
    )

    expected_class_split_counts = {
        (
            brain_class,
            split,
        ): class_split_counter.get(
            (brain_class, split),
            0,
        )
        for brain_class in BrainMRIClass
        for split in DatasetSplit
    }

    actual_class_split_counts = {
        (
            brain_class,
            split,
        ): count
        for brain_class, split, count
        in audit.class_split_counts
    }

    if actual_class_split_counts != expected_class_split_counts:
        raise ValueError(
            "reported class split counts are inconsistent"
        )
