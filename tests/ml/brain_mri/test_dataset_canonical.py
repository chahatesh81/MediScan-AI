from __future__ import annotations

from dataclasses import replace

import pytest

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_canonical import (
    SOURCE_PRIORITY,
    CanonicalCandidate,
    CanonicalDatasetAudit,
    assert_canonical_dataset_integrity,
    build_canonical_dataset,
    canonical_priority_key,
    select_canonical_candidate,
)


def candidate(
    *,
    record_id: str,
    source_id: str = "mendeley_12k",
    content_sha256: str = "a" * 64,
    normalized_class: BrainMRIClass = BrainMRIClass.GLIOMA,
    archive_member: str = "glioma/image.jpg",
) -> CanonicalCandidate:
    return CanonicalCandidate(
        record_id=record_id,
        source_id=source_id,
        archive_name=f"{source_id}.zip",
        archive_member=archive_member,
        normalized_class=normalized_class,
        content_sha256=content_sha256,
    )


def test_source_priority_is_explicit_and_stable() -> None:
    assert SOURCE_PRIORITY == (
        "mendeley_12k",
        "masoud_deduplicated",
        "fernando_30_class",
        "fernando_38_class",
    )


def test_priority_prefers_mendeley() -> None:
    first = candidate(
        record_id="m",
        source_id="mendeley_12k",
    )
    second = candidate(
        record_id="x",
        source_id="masoud_deduplicated",
    )

    assert canonical_priority_key(first) < canonical_priority_key(second)


def test_select_canonical_candidate_uses_source_priority() -> None:
    selected = select_canonical_candidate(
        (
            candidate(
                record_id="masoud",
                source_id="masoud_deduplicated",
            ),
            candidate(
                record_id="mendeley",
                source_id="mendeley_12k",
            ),
        )
    )

    assert selected.record_id == "mendeley"


def test_selection_is_independent_of_input_order() -> None:
    left = candidate(
        record_id="left",
        source_id="fernando_30_class",
    )
    right = candidate(
        record_id="right",
        source_id="mendeley_12k",
    )

    assert (
        select_canonical_candidate((left, right))
        == select_canonical_candidate((right, left))
    )


def test_same_source_tie_breaks_by_member_then_record_id() -> None:
    later = candidate(
        record_id="a",
        archive_member="glioma/z.jpg",
    )
    earlier = candidate(
        record_id="z",
        archive_member="glioma/a.jpg",
    )

    assert (
        select_canonical_candidate((later, earlier))
        == earlier
    )


def test_unknown_source_is_supported_deterministically() -> None:
    selected = select_canonical_candidate(
        (
            candidate(
                record_id="z",
                source_id="unknown_z",
            ),
            candidate(
                record_id="a",
                source_id="unknown_a",
            ),
        )
    )

    assert selected.source_id == "unknown_a"


def test_empty_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty group"):
        select_canonical_candidate(())


def test_mixed_hash_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="one content hash"):
        select_canonical_candidate(
            (
                candidate(
                    record_id="a",
                    content_sha256="a" * 64,
                ),
                candidate(
                    record_id="b",
                    content_sha256="b" * 64,
                ),
            )
        )


def test_class_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        select_canonical_candidate(
            (
                candidate(
                    record_id="a",
                    normalized_class=BrainMRIClass.GLIOMA,
                ),
                candidate(
                    record_id="b",
                    normalized_class=BrainMRIClass.MENINGIOMA,
                ),
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("record_id", "", "record_id"),
        ("source_id", "", "source_id"),
        ("archive_name", "", "archive_name"),
        ("archive_member", "", "archive_member"),
    ),
)
def test_required_text_fields_are_validated(
    field: str,
    value: str,
    message: str,
) -> None:
    item = replace(
        candidate(record_id="record"),
        **{field: value},
    )

    with pytest.raises(ValueError, match=message):
        build_canonical_dataset((item,))


def test_invalid_hash_length_is_rejected() -> None:
    item = candidate(
        record_id="record",
        content_sha256="abc",
    )

    with pytest.raises(ValueError, match="64"):
        build_canonical_dataset((item,))


def test_non_hex_hash_is_rejected() -> None:
    item = candidate(
        record_id="record",
        content_sha256="z" * 64,
    )

    with pytest.raises(ValueError, match="hexadecimal"):
        build_canonical_dataset((item,))


def test_duplicate_record_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate record IDs"):
        build_canonical_dataset(
            (
                candidate(
                    record_id="same",
                    content_sha256="a" * 64,
                ),
                candidate(
                    record_id="same",
                    content_sha256="b" * 64,
                ),
            )
        )


def test_build_canonical_dataset_removes_exact_duplicates() -> None:
    audit = build_canonical_dataset(
        (
            candidate(
                record_id="mendeley",
                source_id="mendeley_12k",
                content_sha256="a" * 64,
            ),
            candidate(
                record_id="masoud",
                source_id="masoud_deduplicated",
                content_sha256="a" * 64,
            ),
            candidate(
                record_id="unique",
                source_id="fernando_30_class",
                content_sha256="b" * 64,
            ),
        )
    )

    assert audit.input_records == 3
    assert audit.unique_content_hashes == 2
    assert audit.duplicate_records_removed == 1

    assert {
        record.record_id
        for record in audit.canonical_records
    } == {
        "mendeley",
        "unique",
    }


def test_duplicate_group_size_is_preserved() -> None:
    audit = build_canonical_dataset(
        (
            candidate(
                record_id="a",
                content_sha256="a" * 64,
            ),
            candidate(
                record_id="b",
                source_id="masoud_deduplicated",
                content_sha256="a" * 64,
            ),
        )
    )

    assert audit.canonical_records[0].duplicate_group_size == 2


def test_output_order_is_deterministic() -> None:
    items = (
        candidate(
            record_id="z",
            content_sha256="b" * 64,
            normalized_class=BrainMRIClass.PITUITARY_TUMOR,
        ),
        candidate(
            record_id="a",
            content_sha256="a" * 64,
            normalized_class=BrainMRIClass.GLIOMA,
        ),
    )

    forward = build_canonical_dataset(items)
    reverse = build_canonical_dataset(reversed(items))

    assert forward == reverse


def test_class_counts_are_reported() -> None:
    audit = build_canonical_dataset(
        (
            candidate(
                record_id="g",
                content_sha256="a" * 64,
                normalized_class=BrainMRIClass.GLIOMA,
            ),
            candidate(
                record_id="m",
                content_sha256="b" * 64,
                normalized_class=BrainMRIClass.MENINGIOMA,
            ),
        )
    )

    assert dict(audit.class_counts) == {
        BrainMRIClass.GLIOMA: 1,
        BrainMRIClass.MENINGIOMA: 1,
    }


def test_source_counts_are_reported() -> None:
    audit = build_canonical_dataset(
        (
            candidate(
                record_id="a",
                content_sha256="a" * 64,
            ),
            candidate(
                record_id="b",
                source_id="masoud_deduplicated",
                content_sha256="b" * 64,
            ),
        )
    )

    assert dict(audit.source_counts) == {
        "masoud_deduplicated": 1,
        "mendeley_12k": 1,
    }


def test_integrity_accepts_valid_audit() -> None:
    audit = build_canonical_dataset(
        (
            candidate(
                record_id="a",
                content_sha256="a" * 64,
            ),
            candidate(
                record_id="b",
                content_sha256="b" * 64,
            ),
        )
    )

    assert_canonical_dataset_integrity(audit)


def test_integrity_rejects_accounting_mismatch() -> None:
    audit = build_canonical_dataset(
        (
            candidate(
                record_id="a",
                content_sha256="a" * 64,
            ),
        )
    )

    broken = replace(
        audit,
        duplicate_records_removed=1,
    )

    with pytest.raises(ValueError, match="accounting"):
        assert_canonical_dataset_integrity(broken)


def test_integrity_rejects_duplicate_content_hashes() -> None:
    audit = build_canonical_dataset(
        (
            candidate(
                record_id="a",
                content_sha256="a" * 64,
            ),
            candidate(
                record_id="b",
                content_sha256="b" * 64,
            ),
        )
    )

    first, second = audit.canonical_records

    broken = CanonicalDatasetAudit(
        input_records=2,
        unique_content_hashes=2,
        duplicate_records_removed=0,
        canonical_records=(
            first,
            replace(
                second,
                content_sha256=first.content_sha256,
            ),
        ),
        class_counts=audit.class_counts,
        source_counts=audit.source_counts,
    )

    with pytest.raises(ValueError, match="duplicate content hashes"):
        assert_canonical_dataset_integrity(broken)
