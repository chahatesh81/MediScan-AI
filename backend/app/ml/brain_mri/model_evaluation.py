"""Production evaluation contracts for Brain MRI classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from backend.app.ml.brain_mri.dataset_loader import (
    BrainMRIClass,
    class_index_mapping,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class BrainMRIEvaluationResult:
    sample_count: int
    accuracy: float
    macro_f1: float
    per_class_metrics: tuple[
        tuple[BrainMRIClass, ClassificationMetrics],
        ...,
    ]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ModelConsistencyResult:
    sample_count: int
    prediction_agreement: float
    maximum_probability_difference: float


def _validate_labels(
    labels: np.ndarray,
    *,
    name: str,
    num_classes: int,
) -> np.ndarray:
    array = np.asarray(labels)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.issubdtype(
        array.dtype,
        np.integer,
    ):
        raise ValueError(
            f"{name} must contain integer class indices."
        )

    normalized = array.astype(
        np.int64,
        copy=False,
    )

    if np.any(normalized < 0):
        raise ValueError(
            f"{name} contains a negative class index."
        )

    if np.any(normalized >= num_classes):
        raise ValueError(
            f"{name} contains an out-of-range class index."
        )

    return normalized


def build_confusion_matrix(
    true_labels: Sequence[int] | np.ndarray,
    predicted_labels: Sequence[int] | np.ndarray,
    *,
    num_classes: int = 4,
) -> np.ndarray:
    if num_classes <= 0:
        raise ValueError(
            "num_classes must be positive."
        )

    true_array = _validate_labels(
        np.asarray(true_labels),
        name="true_labels",
        num_classes=num_classes,
    )

    predicted_array = _validate_labels(
        np.asarray(predicted_labels),
        name="predicted_labels",
        num_classes=num_classes,
    )

    if true_array.shape != predicted_array.shape:
        raise ValueError(
            "true_labels and predicted_labels must "
            "have identical shapes."
        )

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    np.add.at(
        matrix,
        (
            true_array,
            predicted_array,
        ),
        1,
    )

    return matrix


def classification_metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
) -> tuple[ClassificationMetrics, ...]:
    matrix = np.asarray(confusion_matrix)

    if matrix.ndim != 2:
        raise ValueError(
            "confusion_matrix must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "confusion_matrix must be square."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "confusion_matrix must not be empty."
        )

    if np.any(matrix < 0):
        raise ValueError(
            "confusion_matrix must not contain "
            "negative values."
        )

    metrics: list[ClassificationMetrics] = []

    for class_index in range(matrix.shape[0]):
        true_positive = int(
            matrix[class_index, class_index]
        )

        false_positive = int(
            matrix[:, class_index].sum()
            - true_positive
        )

        false_negative = int(
            matrix[class_index, :].sum()
            - true_positive
        )

        support = int(
            matrix[class_index, :].sum()
        )

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
        )

        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )

        f1_denominator = precision + recall

        f1_score = (
            2.0 * precision * recall
            / f1_denominator
            if f1_denominator
            else 0.0
        )

        metrics.append(
            ClassificationMetrics(
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1_score),
                support=support,
            )
        )

    return tuple(metrics)


def evaluate_predictions(
    true_labels: Sequence[int] | np.ndarray,
    predicted_labels: Sequence[int] | np.ndarray,
) -> BrainMRIEvaluationResult:
    mapping = class_index_mapping()
    num_classes = len(mapping)

    matrix = build_confusion_matrix(
        true_labels,
        predicted_labels,
        num_classes=num_classes,
    )

    metrics = (
        classification_metrics_from_confusion_matrix(
            matrix
        )
    )

    sample_count = int(matrix.sum())

    if sample_count <= 0:
        raise ValueError(
            "Evaluation requires at least one sample."
        )

    accuracy = float(
        np.trace(matrix) / sample_count
    )

    macro_f1 = float(
        np.mean(
            [
                metric.f1_score
                for metric in metrics
            ]
        )
    )

    classes_by_index = sorted(
        mapping,
        key=mapping.__getitem__,
    )

    per_class_metrics = tuple(
        zip(
            classes_by_index,
            metrics,
            strict=True,
        )
    )

    return BrainMRIEvaluationResult(
        sample_count=sample_count,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_metrics=per_class_metrics,
        confusion_matrix=tuple(
            tuple(
                int(value)
                for value in row
            )
            for row in matrix
        ),
    )


def probabilities_to_predictions(
    probabilities: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            "probabilities must be a two-dimensional "
            "array."
        )

    if array.shape[0] == 0:
        raise ValueError(
            "probabilities must contain samples."
        )

    if array.shape[1] != len(
        class_index_mapping()
    ):
        raise ValueError(
            "probabilities has an invalid class "
            "dimension."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "probabilities contains non-finite values."
        )

    return np.argmax(
        array,
        axis=1,
    ).astype(np.int64)


def compare_model_probabilities(
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
) -> ModelConsistencyResult:
    first = np.asarray(
        first_probabilities,
        dtype=np.float64,
    )

    second = np.asarray(
        second_probabilities,
        dtype=np.float64,
    )

    if first.shape != second.shape:
        raise ValueError(
            "Model probability arrays must have "
            "identical shapes."
        )

    first_predictions = (
        probabilities_to_predictions(first)
    )

    second_predictions = (
        probabilities_to_predictions(second)
    )

    sample_count = int(first.shape[0])

    prediction_agreement = float(
        np.mean(
            first_predictions
            == second_predictions
        )
    )

    maximum_probability_difference = float(
        np.max(
            np.abs(first - second)
        )
    )

    return ModelConsistencyResult(
        sample_count=sample_count,
        prediction_agreement=prediction_agreement,
        maximum_probability_difference=(
            maximum_probability_difference
        ),
    )


def class_metrics_by_label(
    result: BrainMRIEvaluationResult,
) -> dict[str, ClassificationMetrics]:
    return {
        brain_class.value: metrics
        for brain_class, metrics
        in result.per_class_metrics
    }
