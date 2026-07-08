from __future__ import annotations

import pytest

from backend.app.ml.brain_mri.dataset_deduplication import (
    DeduplicationRecord,
    DuplicateScope,
    LeakageType,
    assert_no_leakage,
    audit_duplicates,
    build_duplicate_groups,
    detect_content_leakage,
    detect_patient_leakage,
    sha256_bytes,
)


def digest(value: str) -> str:
    return sha256_bytes(value.encode())


def record(
    record_id: str,
    source_id: str,
    content: str,
    *,
    patient_id: str | None = None,
    split: str | None = None,
) -> DeduplicationRecord:
    return DeduplicationRecord(
        record_id=record_id,
        source_id=source_id,
        content_sha256=digest(content),
        patient_id=patient_id,
        split=split,
    )


def test_sha256_bytes_is_deterministic() -> None:
    assert sha256_bytes(b"brain-mri") == sha256_bytes(b"brain-mri")


def test_sha256_bytes_changes_with_content() -> None:
    assert sha256_bytes(b"a") != sha256_bytes(b"b")


def test_duplicate_groups_ignore_unique_records() -> None:
    records = (
        record("a", "source_a", "one"),
        record("b", "source_a", "two"),
    )

    assert build_duplicate_groups(records) == ()


def test_within_source_duplicate_is_detected() -> None:
    records = (
        record("a", "source_a", "same"),
        record("b", "source_a", "same"),
    )

    groups = build_duplicate_groups(records)

    assert len(groups) == 1
    assert groups[0].scope is DuplicateScope.WITHIN_SOURCE


def test_cross_source_duplicate_is_detected() -> None:
    records = (
        record("a", "source_a", "same"),
        record("b", "source_b", "same"),
    )

    groups = build_duplicate_groups(records)

    assert len(groups) == 1
    assert groups[0].scope is DuplicateScope.CROSS_SOURCE
    assert groups[0].source_ids == ("source_a", "source_b")


def test_canonical_record_selection_is_deterministic() -> None:
    records = (
        record("z", "source_b", "same"),
        record("a", "source_a", "same"),
        record("m", "source_c", "same"),
    )

    group = build_duplicate_groups(records)[0]

    assert group.canonical_record_id == "a"


def test_input_order_does_not_change_duplicate_groups() -> None:
    first = (
        record("a", "source_a", "same"),
        record("b", "source_b", "same"),
    )

    second = tuple(reversed(first))

    assert build_duplicate_groups(first) == build_duplicate_groups(second)


def test_audit_counts_unique_content() -> None:
    records = (
        record("a", "source_a", "same"),
        record("b", "source_b", "same"),
        record("c", "source_a", "unique"),
    )

    audit = audit_duplicates(records)

    assert audit.total_records == 3
    assert audit.unique_content_hashes == 2


def test_audit_keeps_one_canonical_record_per_hash() -> None:
    records = (
        record("a", "source_a", "same"),
        record("b", "source_b", "same"),
        record("c", "source_a", "unique"),
    )

    audit = audit_duplicates(records)

    assert audit.canonical_record_ids == ("a", "c")
    assert audit.duplicate_record_ids == ("b",)


def test_duplicate_record_id_is_rejected() -> None:
    records = (
        record("same-id", "source_a", "one"),
        record("same-id", "source_b", "two"),
    )

    with pytest.raises(ValueError, match="duplicate record_id"):
        audit_duplicates(records)


def test_invalid_hash_is_rejected() -> None:
    invalid = DeduplicationRecord(
        record_id="a",
        source_id="source_a",
        content_sha256="invalid",
    )

    with pytest.raises(ValueError, match="content_sha256"):
        audit_duplicates((invalid,))


def test_content_in_one_split_is_not_leakage() -> None:
    records = (
        record("a", "source_a", "same", split="train"),
        record("b", "source_b", "same", split="train"),
    )

    assert detect_content_leakage(records) == ()


def test_content_across_splits_is_leakage() -> None:
    records = (
        record("a", "source_a", "same", split="train"),
        record("b", "source_b", "same", split="test"),
    )

    violations = detect_content_leakage(records)

    assert len(violations) == 1
    assert violations[0].leakage_type is LeakageType.CONTENT
    assert violations[0].splits == ("test", "train")


def test_patient_in_one_split_is_not_leakage() -> None:
    records = (
        record(
            "a",
            "source_a",
            "one",
            patient_id="patient-1",
            split="train",
        ),
        record(
            "b",
            "source_a",
            "two",
            patient_id="patient-1",
            split="train",
        ),
    )

    assert detect_patient_leakage(records) == ()


def test_patient_across_splits_is_leakage() -> None:
    records = (
        record(
            "a",
            "source_a",
            "one",
            patient_id="patient-1",
            split="train",
        ),
        record(
            "b",
            "source_a",
            "two",
            patient_id="patient-1",
            split="val",
        ),
    )

    violations = detect_patient_leakage(records)

    assert len(violations) == 1
    assert violations[0].leakage_type is LeakageType.PATIENT
    assert violations[0].key == "patient-1"


def test_missing_patient_id_is_not_patient_leakage() -> None:
    records = (
        record("a", "source_a", "one", split="train"),
        record("b", "source_a", "two", split="test"),
    )

    assert detect_patient_leakage(records) == ()


def test_unassigned_records_are_ignored_by_leakage_checks() -> None:
    records = (
        record("a", "source_a", "same"),
        record("b", "source_b", "same", split="test"),
    )

    assert detect_content_leakage(records) == ()


def test_assert_no_leakage_accepts_safe_records() -> None:
    records = (
        record(
            "a",
            "source_a",
            "one",
            patient_id="patient-1",
            split="train",
        ),
        record(
            "b",
            "source_b",
            "two",
            patient_id="patient-2",
            split="test",
        ),
    )

    assert_no_leakage(records)


def test_assert_no_leakage_rejects_content_leakage() -> None:
    records = (
        record("a", "source_a", "same", split="train"),
        record("b", "source_b", "same", split="test"),
    )

    with pytest.raises(ValueError, match="content=1"):
        assert_no_leakage(records)


def test_assert_no_leakage_rejects_patient_leakage() -> None:
    records = (
        record(
            "a",
            "source_a",
            "one",
            patient_id="patient-1",
            split="train",
        ),
        record(
            "b",
            "source_a",
            "two",
            patient_id="patient-1",
            split="test",
        ),
    )

    with pytest.raises(ValueError, match="patient=1"):
        assert_no_leakage(records)
