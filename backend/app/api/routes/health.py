from fastapi import APIRouter

from backend.app.schemas.prediction import (
    HealthResponse,
)
from backend.app.services.inference_service import (
    inference_service,
)


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    models_loaded = (
        inference_service._v1_model is not None
        and inference_service._v3_model is not None
    )

    return HealthResponse(
        status=(
            "ready"
            if models_loaded
            else "not_ready"
        ),
        service="mediscan-ai",
        models_loaded=models_loaded,
    )
