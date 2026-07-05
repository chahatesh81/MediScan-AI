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
from backend.app.schemas.analysis import (
    AnalysisResponse,
)
from backend.app.services.analysis_service import (
    analysis_service,
)


router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
)
async def analyze(
    file: UploadFile = File(...),
) -> AnalysisResponse:
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
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Combined analysis could "
                "not be completed."
            ),
        ) from exc

    finally:
        await file.close()
