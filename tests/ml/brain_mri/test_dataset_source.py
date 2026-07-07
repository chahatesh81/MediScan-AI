from __future__ import annotations

import pytest

from backend.app.ml.brain_mri.dataset_source import (
    BRAIN_MRI_DATASET_SOURCES,
    BrainMRISourceRole,
    acquisition_sources,
    source_by_id,
)


def test_source_ids_are_unique() -> None:
    source_ids = [
        source.source_id
        for source in BRAIN_MRI_DATASET_SOURCES
    ]

    assert len(source_ids) == len(set(source_ids))


def test_kaggle_refs_are_unique() -> None:
    refs = [
        source.kaggle_ref
        for source in BRAIN_MRI_DATASET_SOURCES
    ]

    assert len(refs) == len(set(refs))


def test_primary_sources_are_exact() -> None:
    primary_ids = {
        source.source_id
        for source in BRAIN_MRI_DATASET_SOURCES
        if source.role is BrainMRISourceRole.PRIMARY
    }

    assert primary_ids == {
        "mendeley_12k",
        "masoud_deduplicated",
    }


def test_original_masoud_family_is_not_counted_twice() -> None:
    selected = acquisition_sources()

    selected_masoud_family = [
        source.source_id
        for source in selected
        if source.source_family
        == "masoud_figshare_sartaj_br35h"
    ]

    assert selected_masoud_family == [
        "masoud_deduplicated"
    ]


def test_patient_aware_sources_require_safe_splitting() -> None:
    patient_aware = [
        source
        for source in BRAIN_MRI_DATASET_SOURCES
        if source.role
        is BrainMRISourceRole.PATIENT_AWARE
    ]

    assert patient_aware

    assert all(
        source.patient_safe_split_required
        for source in patient_aware
    )


def test_rejected_overlap_sources_are_not_acquired() -> None:
    acquired_ids = {
        source.source_id
        for source in acquisition_sources()
    }

    assert "mcnd" not in acquired_ids
    assert "consolidated_masoud_brisc" not in acquired_ids


def test_external_only_source_is_not_acquired() -> None:
    acquired_ids = {
        source.source_id
        for source in acquisition_sources()
    }

    assert "brats_pediatric" not in acquired_ids


def test_acquisition_source_count_is_exact() -> None:
    assert len(acquisition_sources()) == 7


def test_source_lookup_returns_exact_source() -> None:
    source = source_by_id("brats_2021")

    assert source.kaggle_ref == (
        "rayanalhaiek/"
        "brats-2021-preprocessed-npz"
    )


def test_unknown_source_lookup_fails_closed() -> None:
    with pytest.raises(KeyError):
        source_by_id("unknown")
