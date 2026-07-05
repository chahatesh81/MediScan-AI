from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

from backend.app.ml.attention_quality import (
    analyze_attention_quality,
)
from backend.app.ml.gradcam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)
from backend.app.services.inference_service import (
    inference_service,
)


OVERLAY_ALPHA = 0.45


class ExplanationService:
    """
    Grad-CAM explanation service for the frozen V1 primary model.

    Policy:
    - Explain the authoritative V1 classifier only.
    - Reuse the exact V1 production preprocessing path.
    - Do not alter prediction or deployment decisions.
    - Raw heatmaps and visual overlays are presentation artifacts only.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _compute_explanation(
        self,
        image_bytes: bytes,
        *,
        image: np.ndarray | None = None,
        image_batch: Any | None = None,
        probability: float | None = None,
    ) -> dict[str, Any]:
        if not image_bytes:
            raise ValueError(
                "Uploaded image is empty."
            )

        inference_service.load_models()

        model = inference_service._v1_model

        if model is None:
            raise RuntimeError(
                "Primary V1 model is not loaded."
            )

        if image is None:
            image = inference_service.decode_image(
                image_bytes
            )

        if image_batch is None:
            image_batch = (
                inference_service.prepare_v1_input(
                    image_bytes
                )
            )

        if probability is None:
            probability = float(
                model(
                    image_batch,
                    training=False,
                )
                .numpy()
                .reshape(-1)[0]
            )
        else:
            probability = float(probability)

        label = (
            "PNEUMONIA"
            if probability
            >= inference_service.v1_threshold
            else "NORMAL"
        )

        last_conv_layer = (
            find_last_conv_layer(
                model
            )
        )

        with self._lock:
            (
                heatmap,
                explanation_mode,
            ) = generate_gradcam_heatmap(
                model=model,
                image_batch=image_batch,
                last_conv_layer_name=(
                    last_conv_layer
                ),
                return_mode=True,
            )

        if heatmap.ndim != 2:
            raise RuntimeError(
                f"Unexpected Grad-CAM shape: "
                f"{heatmap.shape}"
            )

        if not np.isfinite(heatmap).all():
            raise RuntimeError(
                "Grad-CAM contains invalid values."
            )

        heatmap_min = float(
            np.min(heatmap)
        )

        heatmap_max = float(
            np.max(heatmap)
        )

        if heatmap_min < 0.0:
            raise RuntimeError(
                "Grad-CAM contains negative values."
            )

        if heatmap_max > 1.000001:
            raise RuntimeError(
                "Grad-CAM exceeds normalized range."
            )

        attention_quality = (
            analyze_attention_quality(
                heatmap=heatmap,
                explanation_mode=(
                    explanation_mode
                ),
            )
        )

        heatmap_resized = cv2.resize(
            heatmap,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap_uint8 = np.clip(
            heatmap_resized * 255.0,
            0,
            255,
        ).astype(np.uint8)

        return {
            "image": image,
            "heatmap_uint8": heatmap_uint8,
            "model": "baseline_cnn_v1",
            "prediction": {
                "label": label,
                "probability": probability,
                "threshold": (
                    inference_service.v1_threshold
                ),
            },
            "quality": attention_quality,
            "explanation": {
                "method": "gradcam",
                "mode": explanation_mode,
                "last_conv_layer": (
                    last_conv_layer
                ),
                "raw_heatmap_shape": [
                    int(heatmap.shape[0]),
                    int(heatmap.shape[1]),
                ],
                "output_width": int(
                    image.shape[1]
                ),
                "output_height": int(
                    image.shape[0]
                ),
                "minimum": heatmap_min,
                "maximum": heatmap_max,
            },
        }

    @staticmethod
    def _encode_png(
        image: np.ndarray,
        error_message: str,
    ) -> bytes:
        success, encoded = cv2.imencode(
            ".png",
            image,
        )

        if not success:
            raise RuntimeError(
                error_message
            )

        return encoded.tobytes()

    def explain_bytes(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        result = self._compute_explanation(
            image_bytes
        )

        heatmap_png_bytes = self._encode_png(
            result["heatmap_uint8"],
            "Could not encode Grad-CAM heatmap.",
        )

        return {
            "model": result["model"],
            "prediction": result["prediction"],
            "explanation": result["explanation"],
            "quality": result["quality"],
            "heatmap_png_bytes": (
                heatmap_png_bytes
            ),
        }

    def overlay_bytes(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        result = self._compute_explanation(
            image_bytes
        )

        image = result["image"]
        heatmap_uint8 = result[
            "heatmap_uint8"
        ]

        if image.ndim == 2:
            base_bgr = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        elif (
            image.ndim == 3
            and image.shape[2] == 3
        ):
            base_bgr = image.copy()

        elif (
            image.ndim == 3
            and image.shape[2] == 4
        ):
            base_bgr = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR,
            )

        else:
            raise ValueError(
                "Unsupported image format "
                "for overlay generation."
            )

        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        overlay = cv2.addWeighted(
            base_bgr,
            1.0 - OVERLAY_ALPHA,
            colored_heatmap,
            OVERLAY_ALPHA,
            0.0,
        )

        overlay_png_bytes = self._encode_png(
            overlay,
            "Could not encode Grad-CAM overlay.",
        )

        explanation = dict(
            result["explanation"]
        )

        explanation.update(
            {
                "visualization": (
                    "colored_overlay"
                ),
                "colormap": "jet",
                "overlay_alpha": (
                    OVERLAY_ALPHA
                ),
            }
        )

        return {
            "model": result["model"],
            "prediction": result["prediction"],
            "explanation": explanation,
            "quality": result["quality"],
            "overlay_png_bytes": (
                overlay_png_bytes
            ),
        }


explanation_service = ExplanationService()
