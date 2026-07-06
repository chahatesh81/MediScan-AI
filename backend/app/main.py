from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.modules.runtime_validation import (
    validate_module_runtime,
)
from backend.app.services.inference_service import (
    inference_service,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    validate_module_runtime()
    inference_service.load_models()

    yield


app = FastAPI(
    title="MediScan AI API",
    version="1.0.0",
    description=(
        "Educational chest X-ray decision-support "
        "prototype. Not for clinical use."
    ),
    lifespan=lifespan,
)

app.include_router(
    api_router,
    prefix="/api/v1",
)


@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, str]:
    return {
        "service": "MediScan AI API",
        "status": "running",
        "docs": "/docs",
    }
