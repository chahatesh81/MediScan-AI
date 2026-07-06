from __future__ import annotations

import pytest

from backend.app.modules.execution import (
    ModuleExecutionDecision,
    ModuleNotExecutableError,
    UnknownModuleError,
    authorize_module_execution,
    require_executable_module,
)
from backend.app.modules.registry import (
    BRAIN_TUMOR_MRI,
    BREAST_CANCER,
    CHEST_MULTIDISEASE,
    PNEUMONIA_DETECTION,
    SKIN_DISEASE,
)


def test_require_executable_module_accepts_pneumonia() -> None:
    module = require_executable_module(
        "pneumonia_detection"
    )

    assert module is PNEUMONIA_DETECTION


@pytest.mark.parametrize(
    "module_id",
    [
        "brain_tumor_mri",
        "skin_disease",
        "chest_multidisease",
        "breast_cancer",
    ],
)
def test_require_executable_module_rejects_planned_modules(
    module_id: str,
) -> None:
    with pytest.raises(ModuleNotExecutableError):
        require_executable_module(module_id)


def test_planned_module_error_preserves_policy_context() -> None:
    with pytest.raises(
        ModuleNotExecutableError
    ) as exc_info:
        require_executable_module(
            BRAIN_TUMOR_MRI.module_id
        )

    error = exc_info.value

    assert error.module_id == BRAIN_TUMOR_MRI.module_id
    assert error.status is BRAIN_TUMOR_MRI.status
    assert error.executable is False
    assert str(error) == (
        "Medical module is not executable: "
        "brain_tumor_mri"
    )


def test_require_executable_module_rejects_unknown_module() -> None:
    with pytest.raises(
        UnknownModuleError
    ) as exc_info:
        require_executable_module(
            "unknown_module"
        )

    error = exc_info.value

    assert error.module_id == "unknown_module"
    assert str(error) == (
        "Unknown medical module: unknown_module"
    )


def test_unknown_module_error_preserves_exception_chain() -> None:
    with pytest.raises(
        UnknownModuleError
    ) as exc_info:
        require_executable_module(
            "missing_module"
        )

    assert isinstance(
        exc_info.value.__cause__,
        KeyError,
    )


def test_authorize_module_execution_returns_decision() -> None:
    decision = authorize_module_execution(
        PNEUMONIA_DETECTION.module_id
    )

    assert isinstance(
        decision,
        ModuleExecutionDecision,
    )
    assert decision.module is PNEUMONIA_DETECTION
    assert (
        decision.module_id
        == PNEUMONIA_DETECTION.module_id
    )


def test_execution_decision_is_frozen() -> None:
    decision = authorize_module_execution(
        PNEUMONIA_DETECTION.module_id
    )

    with pytest.raises(
        AttributeError,
    ):
        decision.module = BRAIN_TUMOR_MRI


@pytest.mark.parametrize(
    "module",
    [
        BRAIN_TUMOR_MRI,
        SKIN_DISEASE,
        CHEST_MULTIDISEASE,
        BREAST_CANCER,
    ],
)
def test_authorization_rejects_every_planned_module(
    module,
) -> None:
    with pytest.raises(ModuleNotExecutableError):
        authorize_module_execution(
            module.module_id
        )


def test_authorization_rejects_unknown_module() -> None:
    with pytest.raises(UnknownModuleError):
        authorize_module_execution(
            "not_registered"
        )
