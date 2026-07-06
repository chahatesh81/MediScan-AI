from __future__ import annotations

from unittest.mock import Mock

import pytest

from backend.app.modules import runtime_validation
from backend.app.modules.dispatcher import (
    ModuleExecutorNotRegisteredError,
)
from backend.app.modules.runtime_validation import (
    MissingModuleExecutorError,
    validate_module_runtime,
)


pytestmark = pytest.mark.unit


def test_missing_executor_error_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Mock(
        module_id="pneumonia_detection",
        task_type="binary_classification",
    )

    monkeypatch.setattr(
        runtime_validation,
        "list_executable_modules",
        Mock(return_value=(module,)),
    )

    def missing_executor(
        module_id: str,
    ) -> None:
        raise ModuleExecutorNotRegisteredError(
            module_id
        )

    monkeypatch.setattr(
        runtime_validation,
        "get_module_executor",
        missing_executor,
    )

    with pytest.raises(
        MissingModuleExecutorError
    ) as exc_info:
        validate_module_runtime()

    assert (
        exc_info.value.module_id
        == "pneumonia_detection"
    )
    assert isinstance(
        exc_info.value.__cause__,
        ModuleExecutorNotRegisteredError,
    )


def test_unrelated_runtime_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Mock(
        module_id="pneumonia_detection",
        task_type="binary_classification",
    )

    monkeypatch.setattr(
        runtime_validation,
        "list_executable_modules",
        Mock(return_value=(module,)),
    )

    def fail_executor_lookup(
        module_id: str,
    ) -> None:
        raise RuntimeError(
            "unexpected executor registry failure"
        )

    monkeypatch.setattr(
        runtime_validation,
        "get_module_executor",
        fail_executor_lookup,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected executor registry failure",
    ):
        validate_module_runtime()


def test_normalizer_check_is_not_reached_after_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Mock(
        module_id="pneumonia_detection",
        task_type="binary_classification",
    )

    monkeypatch.setattr(
        runtime_validation,
        "list_executable_modules",
        Mock(return_value=(module,)),
    )

    monkeypatch.setattr(
        runtime_validation,
        "get_module_executor",
        Mock(
            side_effect=(
                ModuleExecutorNotRegisteredError(
                    module.module_id
                )
            )
        ),
    )

    normalizer_mock = Mock(return_value=True)

    monkeypatch.setattr(
        runtime_validation,
        "supports_module_result_normalization",
        normalizer_mock,
    )

    with pytest.raises(
        MissingModuleExecutorError
    ):
        validate_module_runtime()

    normalizer_mock.assert_not_called()
