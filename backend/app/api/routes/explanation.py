from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import Response

from backend.app.services.explanation_service import (
    explanation_service,
)


router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


async def read_validated_image(
    file: UploadFile,
) -> bytes:
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

    return image_bytes


def build_explanation_headers(
    result: dict[str, Any],
    filename: str,
) -> dict[str, str]:
    prediction = result["prediction"]
    explanation = result["explanation"]
    quality = result["quality"]

    headers = {
        "X-Mediscan-Model": result["model"],
        "X-Mediscan-Label": (
            prediction["label"]
        ),
        "X-Mediscan-Probability": (
            f"{prediction['probability']:.12f}"
        ),
        "X-Mediscan-Threshold": (
            f"{prediction['threshold']:.12f}"
        ),
        "X-Mediscan-Explanation-Method": (
            explanation["method"]
        ),
        "X-Mediscan-Explanation-Mode": (
            explanation["mode"]
        ),
        "X-Mediscan-Conv-Layer": (
            explanation["last_conv_layer"]
        ),
        "X-Mediscan-Quality-Status": (
            quality["quality_status"]
        ),
        "X-Mediscan-Display-Warning": (
            str(
                quality["display_warning"]
            ).lower()
        ),
        "X-Mediscan-Warning-Code": (
            quality["warning_code"]
            or "NONE"
        ),
        "X-Mediscan-Border-Energy": (
            f"{quality['border_energy_ratio']:.6f}"
        ),
        "X-Mediscan-Thorax-Energy": (
            f"{quality['thorax_energy_ratio']:.6f}"
        ),
        "X-Mediscan-Peak-In-Border": (
            str(
                bool(
                    quality["peak_in_border"]
                )
            ).lower()
        ),
        "Content-Disposition": (
            "inline; filename="
            + quote(filename)
        ),
    }

    if "visualization" in explanation:
        headers[
            "X-Mediscan-Visualization"
        ] = explanation["visualization"]

    if "colormap" in explanation:
        headers[
            "X-Mediscan-Colormap"
        ] = explanation["colormap"]

    if "overlay_alpha" in explanation:
        headers[
            "X-Mediscan-Overlay-Alpha"
        ] = (
            f"{explanation['overlay_alpha']:.2f}"
        )

    return headers


@router.post(
    "/explain",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": (
                "Raw Grad-CAM heatmap for the "
                "authoritative V1 classifier."
            ),
        }
    },
    tags=["Explanation"],
)
async def explain(
    file: UploadFile = File(...),
) -> Response:
    try:
        image_bytes = await read_validated_image(
            file
        )

        result = (
            explanation_service.explain_bytes(
                image_bytes
            )
        )

        headers = build_explanation_headers(
            result,
            "mediscan_gradcam.png",
        )

        return Response(
            content=result[
                "heatmap_png_bytes"
            ],
            media_type="image/png",
            headers=headers,
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
                "Explanation could not be generated."
            ),
        ) from exc

    finally:
        await file.close()


@router.post(
    "/explain/overlay",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": (
                "Colored Grad-CAM overlay for the "
                "authoritative V1 classifier."
            ),
        }
    },
    tags=["Explanation"],
)
async def explain_overlay(
    file: UploadFile = File(...),
) -> Response:
    try:
        image_bytes = await read_validated_image(
            file
        )

        result = (
            explanation_service.overlay_bytes(
                image_bytes
            )
        )

        headers = build_explanation_headers(
            result,
            "mediscan_gradcam_overlay.png",
        )

        return Response(
            content=result[
                "overlay_png_bytes"
            ],
            media_type="image/png",
            headers=headers,
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
                "Explanation overlay could "
                "not be generated."
            ),
        ) from exc

    finally:
        await file.close()
