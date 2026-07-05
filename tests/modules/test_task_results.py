from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.modules.registry import TaskType
from backend.app.modules.results import (
    BinaryClassificationResult,
    ClassScore,
    MulticlassClassificationResult,
    MultilabelClassificationResult,
    MultilabelClassResult,
)


def test_binary_result_accepts_positive_prediction() -> None:
    result = BinaryClassificationResult(
        predicted_label="PNEUMONIA",
        probability=0.81,
        threshold=0.53,
        negative_label="NORMAL",
        positive_label="PNEUMONIA",
    )

    assert result.task_type is (
        TaskType.BINARY_CLASSIFICATION
    )


def test_binary_threshold_is_inclusive() -> None:
    result = BinaryClassificationResult(
        predicted_label="PNEUMONIA",
        probability=0.53,
        threshold=0.53,
        negative_label="NORMAL",
        positive_label="PNEUMONIA",
    )

    assert result.predicted_label == "PNEUMONIA"


def test_binary_result_rejects_inconsistent_label() -> None:
    with pytest.raises(
        ValidationError,
        match="predicted_label does not match",
    ):
        BinaryClassificationResult(
            predicted_label="NORMAL",
            probability=0.81,
            threshold=0.53,
            negative_label="NORMAL",
            positive_label="PNEUMONIA",
        )


def test_binary_result_rejects_duplicate_labels() -> None:
    with pytest.raises(
        ValidationError,
        match="Binary labels must be distinct",
    ):
        BinaryClassificationResult(
            predicted_label="NORMAL",
            probability=0.20,
            threshold=0.53,
            negative_label="NORMAL",
            positive_label="NORMAL",
        )


def test_probability_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        ClassScore(
            label="Glioma",
            probability=1.01,
        )


def test_multiclass_result_accepts_softmax_distribution() -> None:
    result = MulticlassClassificationResult(
        predicted_label="Glioma",
        scores=(
            ClassScore(
                label="Glioma",
                probability=0.70,
            ),
            ClassScore(
                label="Meningioma",
                probability=0.20,
            ),
            ClassScore(
                label="No Tumor",
                probability=0.10,
            ),
        ),
    )

    assert result.task_type is (
        TaskType.MULTICLASS_CLASSIFICATION
    )


def test_multiclass_result_rejects_non_normalized_scores() -> None:
    with pytest.raises(
        ValidationError,
        match="probabilities must sum to 1",
    ):
        MulticlassClassificationResult(
            predicted_label="Glioma",
            scores=(
                ClassScore(
                    label="Glioma",
                    probability=0.70,
                ),
                ClassScore(
                    label="No Tumor",
                    probability=0.20,
                ),
            ),
        )


def test_multiclass_result_rejects_non_winning_label() -> None:
    with pytest.raises(
        ValidationError,
        match="must have the highest",
    ):
        MulticlassClassificationResult(
            predicted_label="No Tumor",
            scores=(
                ClassScore(
                    label="Glioma",
                    probability=0.80,
                ),
                ClassScore(
                    label="No Tumor",
                    probability=0.20,
                ),
            ),
        )


def test_multiclass_result_rejects_duplicate_labels() -> None:
    with pytest.raises(
        ValidationError,
        match="score labels must be unique",
    ):
        MulticlassClassificationResult(
            predicted_label="Glioma",
            scores=(
                ClassScore(
                    label="Glioma",
                    probability=0.60,
                ),
                ClassScore(
                    label="Glioma",
                    probability=0.40,
                ),
            ),
        )


def test_multilabel_class_accepts_independent_threshold() -> None:
    finding = MultilabelClassResult(
        label="Cardiomegaly",
        probability=0.81,
        threshold=0.70,
        detected=True,
    )

    assert finding.detected is True


def test_multilabel_threshold_is_inclusive() -> None:
    finding = MultilabelClassResult(
        label="Effusion",
        probability=0.67,
        threshold=0.67,
        detected=True,
    )

    assert finding.detected is True


def test_multilabel_class_rejects_inconsistent_state() -> None:
    with pytest.raises(
        ValidationError,
        match="detected does not match",
    ):
        MultilabelClassResult(
            label="Atelectasis",
            probability=0.24,
            threshold=0.50,
            detected=True,
        )


def test_multilabel_result_accepts_independent_findings() -> None:
    result = MultilabelClassificationResult(
        findings=(
            MultilabelClassResult(
                label="Cardiomegaly",
                probability=0.81,
                threshold=0.70,
                detected=True,
            ),
            MultilabelClassResult(
                label="Effusion",
                probability=0.67,
                threshold=0.60,
                detected=True,
            ),
            MultilabelClassResult(
                label="Pneumonia",
                probability=0.11,
                threshold=0.50,
                detected=False,
            ),
        ),
    )

    assert result.task_type is (
        TaskType.MULTILABEL_CLASSIFICATION
    )


def test_multilabel_result_rejects_duplicate_labels() -> None:
    finding = MultilabelClassResult(
        label="Effusion",
        probability=0.67,
        threshold=0.60,
        detected=True,
    )

    with pytest.raises(
        ValidationError,
        match="finding labels must be unique",
    ):
        MultilabelClassificationResult(
            findings=(
                finding,
                finding,
            ),
        )


def test_result_models_are_frozen() -> None:
    result = ClassScore(
        label="Glioma",
        probability=0.70,
    )

    with pytest.raises(ValidationError):
        result.probability = 0.80


def test_result_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ClassScore(
            label="Glioma",
            probability=0.70,
            confidence=0.70,
        )
