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
from backend.app.schemas.analysis import (
    AnalysisResponse,
)
from backend.app.services.analysis_service import (
    analysis_service,
)


router = APIRouter()


@router.post(
    "/modules/{module_id}/analyze",
    response_model=AnalysisResponse,
    tags=["Modules"],
)
async def analyze_module(
    module_id: str,
    file: UploadFile = File(...),
) -> AnalysisResponse:
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

        result = analysis_service.analyze_bytes(
            image_bytes
        )

        return AnalysisResponse(
            **result
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
