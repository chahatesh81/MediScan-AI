from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.app.ml.gradcam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)
from backend.app.services.brain_mri_inference_service import (
    BrainMRIInferenceError,
    brain_mri_inference_service,
)


class BrainMRIExplanationError(RuntimeError):
    """Raised when Brain MRI explanation generation fails."""


class BrainMRIExplanationService:
    """Grad-CAM explanation service for Brain MRI classification."""

    def _compute_explanation(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        if not image_bytes:
            raise ValueError(
                "Uploaded image is empty."
            )

        brain_mri_inference_service.load_model()

        model = brain_mri_inference_service._model

        if model is None:
            raise BrainMRIInferenceError(
                "Brain MRI model failed to load."
            )

        model_input = (
            brain_mri_inference_service.prepare_input(
                image_bytes
            )
        )

        raw_output = model(
            model_input,
            training=False,
        )

        probabilities = np.asarray(
            raw_output.numpy(),
            dtype=np.float64,
        )

        expected_shape = (
            1,
            len(
                brain_mri_inference_service.class_labels
            ),
        )

        if probabilities.shape != expected_shape:
            raise BrainMRIExplanationError(
                "Unexpected Brain MRI explanation "
                f"prediction shape: {probabilities.shape}"
            )

        probability_vector = probabilities[0]

        if not np.all(
            np.isfinite(probability_vector)
        ):
            raise BrainMRIExplanationError(
                "Brain MRI explanation prediction "
                "contains non-finite probabilities."
            )

        predicted_index = int(
            np.argmax(probability_vector)
        )

        predicted_label = (
            brain_mri_inference_service.class_labels[
                predicted_index
            ]
        )

        last_conv_layer = find_last_conv_layer(
            model
        )

        heatmap, explanation_mode = (
            generate_gradcam_heatmap(
                model=model,
                image_batch=model_input,
                last_conv_layer_name=last_conv_layer,
                return_mode=True,
                output_layer_name=(
                    "class_probabilities"
                ),
                target_class_index=predicted_index,
            )
        )

        heatmap_array = np.asarray(
            heatmap,
            dtype=np.float32,
        )

        if heatmap_array.ndim != 2:
            raise BrainMRIExplanationError(
                "Unexpected Grad-CAM shape: "
                f"{heatmap_array.shape}"
            )

        if not np.all(
            np.isfinite(heatmap_array)
        ):
            raise BrainMRIExplanationError(
                "Grad-CAM contains invalid values."
            )

        if np.any(heatmap_array < 0.0):
            raise BrainMRIExplanationError(
                "Grad-CAM contains negative values."
            )

        if np.any(heatmap_array > 1.0):
            raise BrainMRIExplanationError(
                "Grad-CAM exceeds normalized range."
            )

        output_height = int(model_input.shape[1])
        output_width = int(model_input.shape[2])

        resized_heatmap = cv2.resize(
            heatmap_array,
            (
                output_width,
                output_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap_uint8 = np.clip(
            resized_heatmap * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        scores = tuple(
            {
                "label": label,
                "probability": float(
                    probability_vector[index]
                ),
            }
            for index, label in enumerate(
                brain_mri_inference_service.class_labels
            )
        )

        return {
            "model": (
                brain_mri_inference_service
                .model_path.name
            ),
            "prediction": {
                "label": predicted_label,
                "probability": float(
                    probability_vector[
                        predicted_index
                    ]
                ),
                "class_index": predicted_index,
                "scores": scores,
            },
            "explanation": {
                "method": "gradcam",
                "mode": explanation_mode,
                "last_conv_layer": last_conv_layer,
                "target_class_index": (
                    predicted_index
                ),
                "target_class_label": (
                    predicted_label
                ),
                "raw_heatmap_shape": list(
                    heatmap_array.shape
                ),
                "output_width": output_width,
                "output_height": output_height,
                "minimum": float(
                    np.min(heatmap_array)
                ),
                "maximum": float(
                    np.max(heatmap_array)
                ),
            },
            "heatmap_uint8": heatmap_uint8,
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
            raise BrainMRIExplanationError(
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
            "Could not encode Brain MRI "
            "Grad-CAM heatmap.",
        )

        return {
            "model": result["model"],
            "prediction": result["prediction"],
            "explanation": result["explanation"],
            "heatmap_png_bytes": heatmap_png_bytes,
        }


brain_mri_explanation_service = (
    BrainMRIExplanationService()
)
