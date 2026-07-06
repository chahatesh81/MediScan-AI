from __future__ import annotations

from unittest.mock import Mock

import pytest

from backend.app.modules.dispatcher import (
    ModuleExecutorNotRegisteredError,
    dispatch_module_analysis,
    get_module_executor,
)
from backend.app.modules.execution import (
    ModuleNotExecutableError,
    UnknownModuleError,
)
from backend.app.modules import dispatcher


pytestmark = pytest.mark.unit


def test_get_module_executor_returns_pneumonia_executor(
) -> None:
    executor = get_module_executor(
        "pneumonia_detection"
    )

    assert callable(executor)


def test_get_module_executor_rejects_unregistered_executor(
) -> None:
    with pytest.raises(
        ModuleExecutorNotRegisteredError,
        match=(
            "No executor is registered for medical module: "
            "missing_executor"
        ),
    ) as exc_info:
        get_module_executor(
            "missing_executor"
        )

    assert (
        exc_info.value.module_id
        == "missing_executor"
    )


def test_dispatch_executes_pneumonia_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
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
    executor = Mock(
        return_value=expected
    )

    monkeypatch.setitem(
        dispatcher._MODULE_EXECUTORS,
        "pneumonia_detection",
        executor,
    )

    result = dispatch_module_analysis(
        "pneumonia_detection",
        b"image-bytes",
    )

    assert result == expected
    executor.assert_called_once_with(
        b"image-bytes"
    )


@pytest.mark.parametrize(
    "module_id",
    [
        "brain_tumor_mri",
        "skin_disease",
        "chest_multidisease",
        "breast_cancer",
    ],
)
def test_dispatch_rejects_planned_modules_before_resolution(
    module_id: str,
) -> None:
    with pytest.raises(
        ModuleNotExecutableError
    ):
        dispatch_module_analysis(
            module_id,
            b"image-bytes",
        )


def test_dispatch_rejects_unknown_module_before_resolution(
) -> None:
    with pytest.raises(
        UnknownModuleError
    ):
        dispatch_module_analysis(
            "unknown_module",
            b"image-bytes",
        )


def test_dispatch_fails_closed_when_executor_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(
        dispatcher._MODULE_EXECUTORS,
        "pneumonia_detection",
    )

    with pytest.raises(
        ModuleExecutorNotRegisteredError,
        match=(
            "No executor is registered for medical module: "
            "pneumonia_detection"
        ),
    ):
        dispatch_module_analysis(
            "pneumonia_detection",
            b"image-bytes",
        )


def test_pneumonia_adapter_delegates_to_analysis_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "result": "analysis-service-result"
    }
    analyze_mock = Mock(
        return_value=expected
    )

    monkeypatch.setattr(
        dispatcher.analysis_service,
        "analyze_bytes",
        analyze_mock,
    )

    result = (
        dispatcher.execute_pneumonia_detection(
            b"image-bytes"
        )
    )

    assert result == expected
    analyze_mock.assert_called_once_with(
        b"image-bytes"
    )
