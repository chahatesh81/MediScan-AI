from __future__ import annotations

from dataclasses import replace

import pytest

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_canonical import CanonicalRecord
from backend.app.ml.brain_mri.dataset_split import (
    DEFAULT_SPLIT_SEED,
    DatasetSplit,
    DatasetSplitAudit,
    SplitRatios,
    _allocate_class_counts,
    assert_split_integrity,
    build_stratified_split,
    validate_split_ratios,
)


def record(
    *,
    index: int,
    normalized_class: BrainMRIClass = BrainMRIClass.GLIOMA,
    content_sha256: str | None = None,
) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=f"record-{index:05d}",
        source_id="source",
        archive_name="archive.zip",
        archive_member=f"class/image-{index:05d}.jpg",
        normalized_class=normalized_class,
        content_sha256=(
            content_sha256
            if content_sha256 is not None
            else f"{index:064x}"
        ),
        duplicate_group_size=1,
    )


def make_class_records(
    brain_class: BrainMRIClass,
    count: int,
    *,
    offset: int,
) -> tuple[CanonicalRecord, ...]:
    return tuple(
        record(
            index=offset + index,
            normalized_class=brain_class,
        )
        for index in range(count)
    )


def test_default_seed_is_stable() -> None:
    assert DEFAULT_SPLIT_SEED == 42


def test_default_ratios_are_70_15_15() -> None:
    assert SplitRatios() == SplitRatios(
        train=0.70,
        validation=0.15,
        test=0.15,
    )


@pytest.mark.parametrize(
    "ratios",
    (
        SplitRatios(0.0, 0.5, 0.5),
        SplitRatios(-0.1, 0.5, 0.6),
    ),
)
def test_non_positive_ratio_is_rejected(
    ratios: SplitRatios,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        validate_split_ratios(ratios)


@pytest.mark.parametrize(
    "ratios",
    (
        SplitRatios(1.0, 0.0, 0.0),
        SplitRatios(1.1, 0.1, -0.2),
    ),
)
def test_ratio_not_less_than_one_is_rejected(
    ratios: SplitRatios,
) -> None:
    with pytest.raises(ValueError):
        validate_split_ratios(ratios)


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum"):
        validate_split_ratios(
            SplitRatios(
                train=0.60,
                validation=0.20,
                test=0.10,
            )
        )


def test_valid_ratios_are_accepted() -> None:
    validate_split_ratios(
        SplitRatios(
            train=0.70,
            validation=0.15,
            test=0.15,
        )
    )


def test_zero_class_count_allocates_zero() -> None:
    assert _allocate_class_counts(
        0,
        SplitRatios(),
    ) == (0, 0, 0)


def test_negative_class_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        _allocate_class_counts(
            -1,
            SplitRatios(),
        )


@pytest.mark.parametrize(
    ("total", "expected"),
    (
        (1, (1, 0, 0)),
        (2, (2, 0, 0)),
        (3, (2, 1, 0)),
        (4, (3, 1, 0)),
        (10, (7, 2, 1)),
        (20, (14, 3, 3)),
        (100, (70, 15, 15)),
    ),
)
def test_class_allocation_is_deterministic(
    total: int,
    expected: tuple[int, int, int],
) -> None:
    assert _allocate_class_counts(
        total,
        SplitRatios(),
    ) == expected


def test_empty_dataset_is_supported() -> None:
    audit = build_stratified_split(())

    assert audit.total_records == 0
    assert audit.assignments == ()
    assert dict(audit.split_counts) == {
        DatasetSplit.TRAIN: 0,
        DatasetSplit.VALIDATION: 0,
        DatasetSplit.TEST: 0,
    }

    assert_split_integrity(audit)


def test_duplicate_record_ids_are_rejected() -> None:
    first = record(index=1)
    second = replace(
        record(index=2),
        record_id=first.record_id,
    )

    with pytest.raises(ValueError, match="duplicate record IDs"):
        build_stratified_split((first, second))


def test_duplicate_content_hashes_are_rejected() -> None:
    first = record(index=1)
    second = record(
        index=2,
        content_sha256=first.content_sha256,
    )

    with pytest.raises(
        ValueError,
        match="duplicate content hashes",
    ):
        build_stratified_split((first, second))


def test_every_record_is_assigned_once() -> None:
    records = make_class_records(
        BrainMRIClass.GLIOMA,
        100,
        offset=0,
    )

    audit = build_stratified_split(records)

    assert audit.total_records == 100
    assert len(audit.assignments) == 100

    record_ids = [
        assignment.record_id
        for assignment in audit.assignments
    ]

    assert len(record_ids) == len(set(record_ids))


def test_100_records_split_70_15_15() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            100,
            offset=0,
        )
    )

    assert dict(audit.split_counts) == {
        DatasetSplit.TRAIN: 70,
        DatasetSplit.VALIDATION: 15,
        DatasetSplit.TEST: 15,
    }


def test_each_class_is_stratified_independently() -> None:
    records = (
        make_class_records(
            BrainMRIClass.GLIOMA,
            100,
            offset=0,
        )
        + make_class_records(
            BrainMRIClass.MENINGIOMA,
            20,
            offset=1000,
        )
    )

    audit = build_stratified_split(records)

    counts = {
        (
            brain_class,
            split,
        ): count
        for brain_class, split, count
        in audit.class_split_counts
    }

    assert counts[
        (
            BrainMRIClass.GLIOMA,
            DatasetSplit.TRAIN,
        )
    ] == 70

    assert counts[
        (
            BrainMRIClass.GLIOMA,
            DatasetSplit.VALIDATION,
        )
    ] == 15

    assert counts[
        (
            BrainMRIClass.GLIOMA,
            DatasetSplit.TEST,
        )
    ] == 15

    assert counts[
        (
            BrainMRIClass.MENINGIOMA,
            DatasetSplit.TRAIN,
        )
    ] == 14

    assert counts[
        (
            BrainMRIClass.MENINGIOMA,
            DatasetSplit.VALIDATION,
        )
    ] == 3

    assert counts[
        (
            BrainMRIClass.MENINGIOMA,
            DatasetSplit.TEST,
        )
    ] == 3


def test_input_order_does_not_change_assignments() -> None:
    records = make_class_records(
        BrainMRIClass.GLIOMA,
        100,
        offset=0,
    )

    forward = build_stratified_split(records)
    reverse = build_stratified_split(reversed(records))

    assert forward == reverse


def test_same_seed_reproduces_exact_assignments() -> None:
    records = make_class_records(
        BrainMRIClass.GLIOMA,
        100,
        offset=0,
    )

    first = build_stratified_split(
        records,
        seed=123,
    )

    second = build_stratified_split(
        records,
        seed=123,
    )

    assert first == second


def test_different_seed_changes_assignment_membership() -> None:
    records = make_class_records(
        BrainMRIClass.GLIOMA,
        100,
        offset=0,
    )

    first = build_stratified_split(
        records,
        seed=1,
    )

    second = build_stratified_split(
        records,
        seed=2,
    )

    first_mapping = {
        assignment.record_id: assignment.split
        for assignment in first.assignments
    }

    second_mapping = {
        assignment.record_id: assignment.split
        for assignment in second.assignments
    }

    assert first_mapping != second_mapping


def test_custom_ratios_are_supported() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            100,
            offset=0,
        ),
        ratios=SplitRatios(
            train=0.80,
            validation=0.10,
            test=0.10,
        ),
    )

    assert dict(audit.split_counts) == {
        DatasetSplit.TRAIN: 80,
        DatasetSplit.VALIDATION: 10,
        DatasetSplit.TEST: 10,
    }


def test_content_hashes_are_isolated_by_assignment() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            100,
            offset=0,
        )
    )

    hashes_by_split = {
        split: {
            assignment.content_sha256
            for assignment in audit.assignments
            if assignment.split is split
        }
        for split in DatasetSplit
    }

    assert (
        hashes_by_split[DatasetSplit.TRAIN]
        .isdisjoint(
            hashes_by_split[DatasetSplit.VALIDATION]
        )
    )

    assert (
        hashes_by_split[DatasetSplit.TRAIN]
        .isdisjoint(
            hashes_by_split[DatasetSplit.TEST]
        )
    )

    assert (
        hashes_by_split[DatasetSplit.VALIDATION]
        .isdisjoint(
            hashes_by_split[DatasetSplit.TEST]
        )
    )


def test_integrity_accepts_valid_split() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            100,
            offset=0,
        )
    )

    assert_split_integrity(audit)


def test_integrity_rejects_assignment_count_mismatch() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            10,
            offset=0,
        )
    )

    broken = replace(
        audit,
        total_records=11,
    )

    with pytest.raises(
        ValueError,
        match="assignment count",
    ):
        assert_split_integrity(broken)


def test_integrity_rejects_duplicate_assignment_ids() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            10,
            offset=0,
        )
    )

    first = audit.assignments[0]
    second = audit.assignments[1]

    broken = replace(
        audit,
        assignments=(
            first,
            replace(
                second,
                record_id=first.record_id,
            ),
            *audit.assignments[2:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="more than one split",
    ):
        assert_split_integrity(broken)


def test_integrity_rejects_duplicate_assignment_hashes() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            10,
            offset=0,
        )
    )

    first = audit.assignments[0]
    second = audit.assignments[1]

    broken = replace(
        audit,
        assignments=(
            first,
            replace(
                second,
                content_sha256=first.content_sha256,
            ),
            *audit.assignments[2:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="content hashes",
    ):
        assert_split_integrity(broken)


def test_integrity_rejects_incorrect_split_counts() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            100,
            offset=0,
        )
    )

    broken = replace(
        audit,
        split_counts=(
            (DatasetSplit.TRAIN, 100),
            (DatasetSplit.VALIDATION, 0),
            (DatasetSplit.TEST, 0),
        ),
    )

    with pytest.raises(
        ValueError,
        match="split counts",
    ):
        assert_split_integrity(broken)


def test_integrity_rejects_incorrect_class_split_counts() -> None:
    audit = build_stratified_split(
        make_class_records(
            BrainMRIClass.GLIOMA,
            100,
            offset=0,
        )
    )

    broken_counts = list(audit.class_split_counts)

    brain_class, split, count = broken_counts[0]

    broken_counts[0] = (
        brain_class,
        split,
        count + 1,
    )

    broken = replace(
        audit,
        class_split_counts=tuple(broken_counts),
    )

    with pytest.raises(
        ValueError,
        match="class split counts",
    ):
        assert_split_integrity(broken)
