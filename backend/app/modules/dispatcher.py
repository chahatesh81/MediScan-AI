from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.app.modules.execution import (
    authorize_module_execution,
)
from backend.app.services.analysis_service import (
    analysis_service,
)


ModuleExecutor = Callable[
    [bytes],
    dict[str, Any],
]


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


_MODULE_EXECUTORS: dict[
    str,
    ModuleExecutor,
] = {
    "pneumonia_detection": (
        execute_pneumonia_detection
    ),
}


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
) -> dict[str, Any]:
    decision = authorize_module_execution(
        module_id
    )

    executor = get_module_executor(
        decision.module_id
    )

    return executor(
        image_bytes
    )
