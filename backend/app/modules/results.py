from __future__ import annotations

import math

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from backend.app.modules.registry import TaskType


class StrictResultModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class ClassScore(StrictResultModel):
    label: str = Field(min_length=1)
    probability: float = Field(
        ge=0.0,
        le=1.0,
    )


class BinaryClassificationResult(
    StrictResultModel
):
    task_type: TaskType = (
        TaskType.BINARY_CLASSIFICATION
    )
    predicted_label: str = Field(min_length=1)
    probability: float = Field(
        ge=0.0,
        le=1.0,
    )
    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )
    negative_label: str = Field(min_length=1)
    positive_label: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_binary_result(
        self,
    ) -> BinaryClassificationResult:
        if self.negative_label == self.positive_label:
            raise ValueError(
                "Binary labels must be distinct."
            )

        expected_label = (
            self.positive_label
            if self.probability >= self.threshold
            else self.negative_label
        )

        if self.predicted_label != expected_label:
            raise ValueError(
                "predicted_label does not match "
                "probability and threshold."
            )

        return self


class MulticlassClassificationResult(
    StrictResultModel
):
    task_type: TaskType = (
        TaskType.MULTICLASS_CLASSIFICATION
    )
    predicted_label: str = Field(min_length=1)
    scores: tuple[ClassScore, ...]

    @model_validator(mode="after")
    def validate_multiclass_result(
        self,
    ) -> MulticlassClassificationResult:
        if len(self.scores) < 2:
            raise ValueError(
                "Multiclass results require at least "
                "two class scores."
            )

        labels = tuple(
            score.label
            for score in self.scores
        )

        if len(set(labels)) != len(labels):
            raise ValueError(
                "Multiclass score labels must be unique."
            )

        probability_sum = sum(
            score.probability
            for score in self.scores
        )

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Multiclass probabilities must sum to 1."
            )

        maximum_probability = max(
            score.probability
            for score in self.scores
        )

        winning_labels = {
            score.label
            for score in self.scores
            if math.isclose(
                score.probability,
                maximum_probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        }

        if self.predicted_label not in winning_labels:
            raise ValueError(
                "predicted_label must have the highest "
                "multiclass probability."
            )

        return self


class MultilabelClassResult(
    StrictResultModel
):
    label: str = Field(min_length=1)
    probability: float = Field(
        ge=0.0,
        le=1.0,
    )
    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )
    detected: bool

    @model_validator(mode="after")
    def validate_detection_state(
        self,
    ) -> MultilabelClassResult:
        expected_detection = (
            self.probability >= self.threshold
        )

        if self.detected is not expected_detection:
            raise ValueError(
                "detected does not match probability "
                "and threshold."
            )

        return self


class MultilabelClassificationResult(
    StrictResultModel
):
    task_type: TaskType = (
        TaskType.MULTILABEL_CLASSIFICATION
    )
    findings: tuple[MultilabelClassResult, ...]

    @model_validator(mode="after")
    def validate_multilabel_result(
        self,
    ) -> MultilabelClassificationResult:
        if not self.findings:
            raise ValueError(
                "Multilabel results require at least "
                "one finding."
            )

        labels = tuple(
            finding.label
            for finding in self.findings
        )

        if len(set(labels)) != len(labels):
            raise ValueError(
                "Multilabel finding labels must be unique."
            )

        return self
