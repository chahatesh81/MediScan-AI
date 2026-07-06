from __future__ import annotations

from dataclasses import dataclass

from backend.app.modules.dispatcher import (
    ModuleExecutorNotRegisteredError,
    get_module_executor,
)
from backend.app.modules.normalization import (
    supports_module_result_normalization,
)
from backend.app.modules.registry import (
    MedicalModule,
    list_executable_modules,
)


class ModuleRuntimeValidationError(RuntimeError):
    """Base error for invalid module runtime configuration."""


class MissingModuleExecutorError(
    ModuleRuntimeValidationError
):
    """Raised when an executable module has no executor."""

    def __init__(
        self,
        module: MedicalModule,
    ) -> None:
        self.module_id = module.module_id
        super().__init__(
            "Executable medical module has no registered "
            f"executor: {module.module_id}"
        )


class MissingModuleNormalizerError(
    ModuleRuntimeValidationError
):
    """Raised when an executable module has no normalizer."""

    def __init__(
        self,
        module: MedicalModule,
    ) -> None:
        self.module_id = module.module_id
        self.task_type = module.task_type
        super().__init__(
            "Executable medical module has no result "
            f"normalizer: {module.module_id}"
        )


@dataclass(frozen=True, slots=True)
class ModuleRuntimeValidationResult:
    validated_module_ids: tuple[str, ...]


def validate_module_runtime(
) -> ModuleRuntimeValidationResult:
    validated_module_ids: list[str] = []

    for module in list_executable_modules():
        try:
            get_module_executor(module.module_id)
        except ModuleExecutorNotRegisteredError as exc:
            raise MissingModuleExecutorError(
                module
            ) from exc

        if not supports_module_result_normalization(
            module
        ):
            raise MissingModuleNormalizerError(
                module
            )

        validated_module_ids.append(
            module.module_id
        )

    return ModuleRuntimeValidationResult(
        validated_module_ids=tuple(
            validated_module_ids
        )
    )
