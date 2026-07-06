from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.modules.normalization import (
    UnsupportedResultNormalizationError,
    normalize_module_result,
    normalize_pneumonia_result,
)
from backend.app.modules.registry import (
    get_module,
)
from backend.app.modules.results import (
    BinaryClassificationResult,
)


def make_pneumonia_payload(
    *,
    label: str = "PNEUMONIA",
    probability: float = 0.81,
    threshold: float = 0.53,
) -> dict[str, object]:
    return {
        "primary_prediction": {
            "model": "baseline_cnn_v1",
            "label": label,
            "probability": probability,
            "threshold": threshold,
        },
        "secondary_signal": {
            "model": "advanced_v3",
        },
    }


def test_pneumonia_normalizer_returns_binary_result(
) -> None:
    result = normalize_pneumonia_result(
        make_pneumonia_payload()
    )

    assert isinstance(
        result,
        BinaryClassificationResult,
    )
    assert result.predicted_label == "PNEUMONIA"
    assert result.probability == 0.81
    assert result.threshold == 0.53
    assert result.negative_label == "NORMAL"
    assert result.positive_label == "PNEUMONIA"


def test_pneumonia_normalizer_accepts_negative_result(
) -> None:
    result = normalize_pneumonia_result(
        make_pneumonia_payload(
            label="NORMAL",
            probability=0.20,
        )
    )

    assert result.predicted_label == "NORMAL"


def test_pneumonia_normalizer_preserves_threshold_boundary(
) -> None:
    result = normalize_pneumonia_result(
        make_pneumonia_payload(
            label="PNEUMONIA",
            probability=0.53,
            threshold=0.53,
        )
    )

    assert result.predicted_label == "PNEUMONIA"


def test_pneumonia_normalizer_rejects_inconsistent_label(
) -> None:
    with pytest.raises(
        ValidationError,
        match="predicted_label does not match",
    ):
        normalize_pneumonia_result(
            make_pneumonia_payload(
                label="NORMAL",
                probability=0.81,
            )
        )


def test_pneumonia_normalizer_requires_primary_prediction(
) -> None:
    with pytest.raises(KeyError):
        normalize_pneumonia_result({})


def test_pneumonia_normalizer_requires_label(
) -> None:
    payload = make_pneumonia_payload()
    primary = payload["primary_prediction"]
    assert isinstance(primary, dict)
    del primary["label"]

    with pytest.raises(KeyError):
        normalize_pneumonia_result(payload)


def test_module_normalizer_dispatches_pneumonia(
) -> None:
    module = get_module("pneumonia_detection")

    result = normalize_module_result(
        module,
        make_pneumonia_payload(),
    )

    assert isinstance(
        result,
        BinaryClassificationResult,
    )
    assert result.predicted_label == "PNEUMONIA"


@pytest.mark.parametrize(
    "module_id",
    [
        "brain_tumor_mri",
        "skin_disease",
        "chest_multidisease",
        "breast_cancer",
    ],
)
def test_module_normalizer_rejects_unsupported_modules(
    module_id: str,
) -> None:
    module = get_module(module_id)

    with pytest.raises(
        UnsupportedResultNormalizationError,
        match=module_id,
    ):
        normalize_module_result(
            module,
            {},
        )
