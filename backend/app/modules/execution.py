from __future__ import annotations

from dataclasses import dataclass

from backend.app.modules.registry import (
    MedicalModule,
    ModuleStatus,
    get_module,
)


class ModuleExecutionError(RuntimeError):
    """Base error for module execution policy failures."""


class UnknownModuleError(ModuleExecutionError):
    """Raised when execution targets an unregistered module."""

    def __init__(
        self,
        module_id: str,
    ) -> None:
        self.module_id = module_id
        super().__init__(
            f"Unknown medical module: {module_id}"
        )


class ModuleNotExecutableError(ModuleExecutionError):
    """Raised when a registered module may not execute."""

    def __init__(
        self,
        module: MedicalModule,
    ) -> None:
        self.module_id = module.module_id
        self.status = module.status
        self.executable = module.executable
        super().__init__(
            "Medical module is not executable: "
            f"{module.module_id}"
        )


@dataclass(frozen=True, slots=True)
class ModuleExecutionDecision:
    module: MedicalModule

    @property
    def module_id(self) -> str:
        return self.module.module_id


def require_executable_module(
    module_id: str,
) -> MedicalModule:
    try:
        module = get_module(module_id)
    except KeyError as exc:
        raise UnknownModuleError(
            module_id
        ) from exc

    if module.status is not ModuleStatus.AVAILABLE:
        raise ModuleNotExecutableError(module)

    if not module.executable:
        raise ModuleNotExecutableError(module)

    return module


def authorize_module_execution(
    module_id: str,
) -> ModuleExecutionDecision:
    module = require_executable_module(module_id)

    return ModuleExecutionDecision(
        module=module,
    )
