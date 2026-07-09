from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.app.modules.execution import (
    authorize_module_execution,
)
from backend.app.modules.normalization import (
    NormalizedModuleResult,
    normalize_module_result,
)
from backend.app.modules.registry import (
    MedicalModule,
    get_module,
)
from backend.app.services.analysis_service import (
    analysis_service,
)
from backend.app.services.brain_mri_inference_service import (
    brain_mri_inference_service,
)


ModuleExecutor = Callable[
    [bytes],
    dict[str, Any],
]


@dataclass(frozen=True, slots=True)
class ModuleDispatchResult:
    module: MedicalModule
    result: NormalizedModuleResult


class ModuleExecutorRegistrationError(
    RuntimeError
):
    """Base error for executor registration failures."""


class UnknownModuleExecutorRegistrationError(
    ModuleExecutorRegistrationError
):
    """Raised when registration targets an unknown module."""

    def __init__(
        self,
        module_id: str,
    ) -> None:
        self.module_id = module_id
        super().__init__(
            "Cannot register executor for unknown "
            f"medical module: {module_id}"
        )


class ModuleExecutorAlreadyRegisteredError(
    ModuleExecutorRegistrationError
):
    """Raised when a module already has an executor."""

    def __init__(
        self,
        module_id: str,
    ) -> None:
        self.module_id = module_id
        super().__init__(
            "An executor is already registered for "
            f"medical module: {module_id}"
        )


class ModuleExecutorNotRegisteredError(
    RuntimeError
):
    """Raised when an executable module has no executor."""

    def __init__(
        self,
        module_id: str,
    ) -> None:
        self.module_id = module_id
        super().__init__(
            "No executor is registered for medical module: "
            f"{module_id}"
        )


def execute_pneumonia_detection(
    image_bytes: bytes,
) -> dict[str, Any]:
    return analysis_service.analyze_bytes(
        image_bytes
    )


def execute_brain_tumor_mri(
    image_bytes: bytes,
) -> dict[str, Any]:
    return brain_mri_inference_service.predict_bytes(
        image_bytes
    )


_MODULE_EXECUTORS: dict[
    str,
    ModuleExecutor,
] = {}


def register_module_executor(
    module_id: str,
    executor: ModuleExecutor,
) -> None:
    try:
        get_module(module_id)
    except KeyError as exc:
        raise UnknownModuleExecutorRegistrationError(
            module_id
        ) from exc

    if module_id in _MODULE_EXECUTORS:
        raise ModuleExecutorAlreadyRegisteredError(
            module_id
        )

    _MODULE_EXECUTORS[module_id] = executor


def get_module_executor(
    module_id: str,
) -> ModuleExecutor:
    try:
        return _MODULE_EXECUTORS[module_id]
    except KeyError as exc:
        raise ModuleExecutorNotRegisteredError(
            module_id
        ) from exc


def dispatch_module_analysis(
    module_id: str,
    image_bytes: bytes,
) -> ModuleDispatchResult:
    decision = authorize_module_execution(
        module_id
    )

    executor = get_module_executor(
        decision.module_id
    )

    payload = executor(
        image_bytes
    )

    result = normalize_module_result(
        decision.module,
        payload,
    )

    return ModuleDispatchResult(
        module=decision.module,
        result=result,
    )


register_module_executor(
    "pneumonia_detection",
    execute_pneumonia_detection,
)

register_module_executor(
    "brain_tumor_mri",
    execute_brain_tumor_mri,
)
