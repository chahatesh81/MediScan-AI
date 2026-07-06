from __future__ import annotations

import pytest

from backend.app.modules import runtime_validation
from backend.app.modules.registry import (
    BRAIN_TUMOR_MRI,
    PNEUMONIA_DETECTION,
)
from backend.app.modules.dispatcher import (
    ModuleExecutorNotRegisteredError,
)
from backend.app.modules.runtime_validation import (
    MissingModuleExecutorError,
    MissingModuleNormalizerError,
    ModuleRuntimeValidationResult,
    validate_module_runtime,
)


pytestmark = pytest.mark.unit


def test_runtime_validation_accepts_current_configuration(
) -> None:
    result = validate_module_runtime()

    assert isinstance(
        result,
        ModuleRuntimeValidationResult,
    )
    assert result.validated_module_ids == (
        "pneumonia_detection",
    )


def test_runtime_validation_rejects_missing_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_executor(
        module_id: str,
    ):
        raise ModuleExecutorNotRegisteredError(
            module_id
        )

    monkeypatch.setattr(
        runtime_validation,
        "get_module_executor",
        missing_executor,
    )

    with pytest.raises(
        MissingModuleExecutorError,
        match="pneumonia_detection",
    ) as exc_info:
        validate_module_runtime()

    assert (
        exc_info.value.module_id
        == PNEUMONIA_DETECTION.module_id
    )
    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )


def test_runtime_validation_rejects_missing_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_validation,
        "supports_module_result_normalization",
        lambda module: False,
    )

    with pytest.raises(
        MissingModuleNormalizerError,
        match="pneumonia_detection",
    ) as exc_info:
        validate_module_runtime()

    assert (
        exc_info.value.module_id
        == PNEUMONIA_DETECTION.module_id
    )
    assert (
        exc_info.value.task_type
        is PNEUMONIA_DETECTION.task_type
    )


def test_runtime_validation_checks_only_executable_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_module_ids: list[str] = []

    def record_normalizer_check(
        module,
    ) -> bool:
        checked_module_ids.append(
            module.module_id
        )
        return True

    monkeypatch.setattr(
        runtime_validation,
        "supports_module_result_normalization",
        record_normalizer_check,
    )

    result = validate_module_runtime()

    assert checked_module_ids == [
        "pneumonia_detection"
    ]
    assert (
        BRAIN_TUMOR_MRI.module_id
        not in checked_module_ids
    )
    assert result.validated_module_ids == (
        "pneumonia_detection",
    )


def test_runtime_validation_result_is_frozen(
) -> None:
    result = validate_module_runtime()

    with pytest.raises(AttributeError):
        result.validated_module_ids = ()
