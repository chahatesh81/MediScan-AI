from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.prediction import (
    Decision,
    PrimaryPrediction,
    SecondarySignal,
)


pytestmark = pytest.mark.unit


def test_primary_prediction_accepts_frozen_model() -> None:
    prediction = PrimaryPrediction(
        model="baseline_cnn_v1",
        label="PNEUMONIA",
        probability=0.91,
        threshold=0.53,
    )

    assert prediction.model == "baseline_cnn_v1"
    assert prediction.label == "PNEUMONIA"


def test_primary_prediction_rejects_unknown_model() -> None:
    with pytest.raises(ValidationError):
        PrimaryPrediction(
            model="unknown_model",
            label="NORMAL",
            probability=0.2,
            threshold=0.53,
        )


def test_secondary_signal_preserves_exploratory_role() -> None:
    signal = SecondarySignal(
        model="advanced_v3",
        role="exploratory",
        probability=0.8,
        threshold=0.5,
        predicted_label="PNEUMONIA",
        automatic_override_allowed=False,
    )

    assert signal.role == "exploratory"
    assert signal.automatic_override_allowed is False


def test_decision_rejects_unsupported_warning_code() -> None:
    with pytest.raises(ValidationError):
        Decision(
            final_label="NORMAL",
            source="baseline_cnn_v1",
            manual_review_recommended=True,
            warning_code="UNSUPPORTED_WARNING",
        )
