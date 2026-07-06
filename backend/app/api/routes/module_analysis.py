from __future__ import annotations

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)

from backend.app.api.routes.explanation import (
    read_validated_image,
)
from backend.app.modules.execution import (
    ModuleNotExecutableError,
    UnknownModuleError,
    authorize_module_execution,
)
from backend.app.modules.dispatcher import (
    dispatch_module_analysis,
)
from backend.app.modules.responses import (
    ModuleAnalysisResponse,
)


router = APIRouter()


@router.post(
    "/modules/{module_id}/analyze",
    response_model=ModuleAnalysisResponse,
    tags=["Modules"],
)
async def analyze_module(
    module_id: str,
    file: UploadFile = File(...),
) -> ModuleAnalysisResponse:
    try:
        authorize_module_execution(module_id)
    except UnknownModuleError as exc:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ModuleNotExecutableError as exc:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    try:
        image_bytes = await read_validated_image(
            file
        )

        dispatch_result = dispatch_module_analysis(
            module_id,
            image_bytes,
        )
        module = dispatch_result.module

        return ModuleAnalysisResponse(
            module_id=module.module_id,
            display_name=module.display_name,
            modality=module.modality,
            task_type=module.task_type,
            result=dispatch_result.result,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Module analysis could not "
                "be completed."
            ),
        ) from exc

    finally:
        await file.close()
