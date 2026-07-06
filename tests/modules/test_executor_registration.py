from __future__ import annotations

from unittest.mock import Mock

import pytest

from backend.app.modules import dispatcher
from backend.app.modules.dispatcher import (
    ModuleExecutorAlreadyRegisteredError,
    UnknownModuleExecutorRegistrationError,
    get_module_executor,
    register_module_executor,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def isolated_executor_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dispatcher.ModuleExecutor]:
    registry: dict[
        str,
        dispatcher.ModuleExecutor,
    ] = {}

    monkeypatch.setattr(
        dispatcher,
        "_MODULE_EXECUTORS",
        registry,
    )

    return registry


def test_register_module_executor_registers_known_module(
    isolated_executor_registry: dict[
        str,
        dispatcher.ModuleExecutor,
    ],
) -> None:
    executor = Mock()

    register_module_executor(
        "pneumonia_detection",
        executor,
    )

    assert get_module_executor(
        "pneumonia_detection"
    ) is executor

    assert isolated_executor_registry == {
        "pneumonia_detection": executor
    }


def test_register_module_executor_rejects_unknown_module(
    isolated_executor_registry: dict[
        str,
        dispatcher.ModuleExecutor,
    ],
) -> None:
    executor = Mock()

    with pytest.raises(
        UnknownModuleExecutorRegistrationError,
        match=(
            "Cannot register executor for unknown "
            "medical module: unknown_module"
        ),
    ) as exc_info:
        register_module_executor(
            "unknown_module",
            executor,
        )

    assert exc_info.value.module_id == "unknown_module"
    assert isolated_executor_registry == {}


def test_register_module_executor_rejects_duplicate(
    isolated_executor_registry: dict[
        str,
        dispatcher.ModuleExecutor,
    ],
) -> None:
    first_executor = Mock()
    second_executor = Mock()

    register_module_executor(
        "pneumonia_detection",
        first_executor,
    )

    with pytest.raises(
        ModuleExecutorAlreadyRegisteredError,
        match=(
            "An executor is already registered for "
            "medical module: pneumonia_detection"
        ),
    ) as exc_info:
        register_module_executor(
            "pneumonia_detection",
            second_executor,
        )

    assert (
        exc_info.value.module_id
        == "pneumonia_detection"
    )
    assert get_module_executor(
        "pneumonia_detection"
    ) is first_executor


@pytest.mark.parametrize(
    "module_id",
    [
        "brain_tumor_mri",
        "skin_disease",
        "chest_multidisease",
        "breast_cancer",
    ],
)
def test_registration_accepts_known_planned_modules(
    module_id: str,
    isolated_executor_registry: dict[
        str,
        dispatcher.ModuleExecutor,
    ],
) -> None:
    executor = Mock()

    register_module_executor(
        module_id,
        executor,
    )

    assert get_module_executor(module_id) is executor


def test_default_pneumonia_executor_is_registered(
) -> None:
    executor = get_module_executor(
        "pneumonia_detection"
    )

    assert (
        executor
        is dispatcher.execute_pneumonia_detection
    )
