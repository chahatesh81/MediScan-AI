from __future__ import annotations

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)

from backend.app.schemas.prediction import (
    PredictionResponse,
)
from backend.app.services.inference_service import (
    inference_service,
)


router = APIRouter()


MAX_UPLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict(
    file: UploadFile = File(...),
) -> PredictionResponse:
    if file.content_type not in (
        ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail=(
                "Unsupported image type. "
                "Use JPEG or PNG."
            ),
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Uploaded image is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=(
                status.HTTP_413_CONTENT_TOO_LARGE
            ),
            detail=(
                "Image exceeds the 10 MB "
                "upload limit."
            ),
        )

    try:
        result = (
            inference_service.predict_bytes(
                image_bytes
            )
        )

        return PredictionResponse(
            **result
        )

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
                "Prediction could not be completed."
            ),
        ) from exc

    finally:
        await file.close()
