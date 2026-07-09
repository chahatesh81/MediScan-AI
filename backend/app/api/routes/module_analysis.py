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
)
from backend.app.modules.dispatcher import (
    dispatch_module_analysis,
)
from backend.app.modules.responses import (
    ModuleAnalysisResponse,
)
from backend.app.services.brain_mri_explanation_service import (
    brain_mri_explanation_service,
)


router = APIRouter()


@router.post(
    "/modules/brain_tumor_mri/explain",
    tags=["Modules"],
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    },
)
async def explain_brain_tumor_mri(
    file: UploadFile = File(...),
):
    try:
        image_bytes = await read_validated_image(
            file
        )

        result = (
            brain_mri_explanation_service
            .explain_bytes(image_bytes)
        )

        from fastapi.responses import Response

        return Response(
            content=result["heatmap_png_bytes"],
            media_type="image/png",
            headers={
                "X-MediScan-Module": (
                    "brain_tumor_mri"
                ),
                "X-MediScan-Predicted-Label": (
                    result["prediction"]["label"]
                ),
                "X-MediScan-Explanation-Method": (
                    result["explanation"]["method"]
                ),
            },
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
                "Brain MRI explanation could not "
                "be completed."
            ),
        ) from exc

    finally:
        await file.close()


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

    except UnknownModuleError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ModuleNotExecutableError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

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
