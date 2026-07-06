from __future__ import annotations

from unittest.mock import Mock

import pytest

from backend.app.modules import dispatcher
from backend.app.modules.dispatcher import (
    ModuleDispatchResult,
    dispatch_module_analysis,
)
from backend.app.modules.results import (
    BinaryClassificationResult,
)


pytestmark = pytest.mark.unit


def pneumonia_payload() -> dict[str, object]:
    return {
        "primary_prediction": {
            "model": "baseline_cnn_v1",
            "label": "PNEUMONIA",
            "probability": 0.81,
            "threshold": 0.53,
        },
        "secondary_signal": {
            "model": "advanced_v3",
        },
    }


def test_dispatch_returns_execution_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Mock(
        return_value=pneumonia_payload()
    )

    monkeypatch.setitem(
        dispatcher._MODULE_EXECUTORS,
        "pneumonia_detection",
        executor,
    )

    dispatch_result = dispatch_module_analysis(
        "pneumonia_detection",
        b"image-bytes",
    )

    assert isinstance(
        dispatch_result,
        ModuleDispatchResult,
    )
    assert (
        dispatch_result.module.module_id
        == "pneumonia_detection"
    )
    assert isinstance(
        dispatch_result.result,
        BinaryClassificationResult,
    )


def test_dispatch_envelope_preserves_module_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        dispatcher._MODULE_EXECUTORS,
        "pneumonia_detection",
        Mock(
            return_value=pneumonia_payload()
        ),
    )

    dispatch_result = dispatch_module_analysis(
        "pneumonia_detection",
        b"image-bytes",
    )

    assert (
        dispatch_result.module.display_name
        == "Pneumonia Detection"
    )
    assert dispatch_result.module.executable is True


def test_dispatch_envelope_is_frozen() -> None:
    from backend.app.modules.registry import get_module

    envelope = ModuleDispatchResult(
        module=get_module(
            "pneumonia_detection"
        ),
        result=BinaryClassificationResult(
            predicted_label="PNEUMONIA",
            probability=0.81,
            threshold=0.53,
            negative_label="NORMAL",
            positive_label="PNEUMONIA",
        ),
    )

    with pytest.raises(
        AttributeError,
    ):
        envelope.result = envelope.result
