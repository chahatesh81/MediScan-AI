from fastapi import APIRouter

from backend.app.api.routes.analysis import (
    router as analysis_router,
)
from backend.app.api.routes.explanation import (
    router as explanation_router,
)
from backend.app.api.routes.health import (
    router as health_router,
)
from backend.app.api.routes.model_info import (
    router as model_info_router,
)
from backend.app.api.routes.module_analysis import (
    router as module_analysis_router,
)
from backend.app.api.routes.modules import (
    router as modules_router,
)
from backend.app.api.routes.prediction import (
    router as prediction_router,
)


api_router = APIRouter()

api_router.include_router(
    health_router
)

api_router.include_router(
    model_info_router
)

api_router.include_router(
    module_analysis_router
)

api_router.include_router(
    modules_router
)

api_router.include_router(
    prediction_router
)

api_router.include_router(
    explanation_router
)

api_router.include_router(
    analysis_router
)
