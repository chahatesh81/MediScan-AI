from __future__ import annotations

from typing import Any

from backend.app.services.explanation_service import (
    explanation_service,
)
from backend.app.services.inference_service import (
    inference_service,
)


class AnalysisService:
    """
    Combined production analysis orchestration.

    Policy:
    - V1 remains the authoritative primary classifier.
    - V3 remains an exploratory secondary safety signal.
    - V3 never automatically overrides V1.
    - Explanation quality never changes the diagnosis.
    - Combined analysis reuses one V1 forward pass.
    """

    def analyze_bytes(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        inference_service.load_models()

        v1_model = inference_service._v1_model
        v3_model = inference_service._v3_model

        if (
            v1_model is None
            or v3_model is None
        ):
            raise RuntimeError(
                "Models failed to load."
            )

        image = inference_service.decode_image(
            image_bytes
        )

        v1_input = (
            inference_service.prepare_v1_input(
                image_bytes
            )
        )

        (
            v3_input,
            preprocessing_metadata,
        ) = inference_service.prepare_v3_input(
            image
        )

        v1_probability = float(
            v1_model(
                v1_input,
                training=False,
            )
            .numpy()
            .reshape(-1)[0]
        )

        v3_probability = float(
            v3_model(
                v3_input,
                training=False,
            )
            .numpy()
            .reshape(-1)[0]
        )

        v1_positive = (
            v1_probability
            >= inference_service.v1_threshold
        )

        v3_positive = (
            v3_probability
            >= inference_service.v3_threshold
        )

        primary_label = (
            "PNEUMONIA"
            if v1_positive
            else "NORMAL"
        )

        manual_review_warning = (
            not v1_positive
            and v3_positive
        )

        prediction = {
            "primary_prediction": {
                "model": "baseline_cnn_v1",
                "label": primary_label,
                "probability": v1_probability,
                "threshold": (
                    inference_service.v1_threshold
                ),
            },
            "secondary_signal": {
                "model": "advanced_v3",
                "role": "exploratory",
                "probability": v3_probability,
                "threshold": (
                    inference_service.v3_threshold
                ),
                "predicted_label": (
                    "PNEUMONIA"
                    if v3_positive
                    else "NORMAL"
                ),
                "automatic_override_allowed": False,
            },
            "decision": {
                "final_label": primary_label,
                "source": "baseline_cnn_v1",
                "manual_review_recommended": (
                    manual_review_warning
                ),
                "warning_code": (
                    "V1_NORMAL_V3_PNEUMONIA"
                    if manual_review_warning
                    else None
                ),
            },
            "preprocessing": {
                "v1": (
                    "rgb_bilinear_resize_224"
                ),
                "v3": (
                    "artifact_aware_preprocess_xray"
                ),
                "v3_metadata": (
                    preprocessing_metadata
                ),
            },
            "disclaimer": (
                "Educational decision-support prototype. "
                "Not for clinical use. Human review required."
            ),
        }

        explanation_result = (
            explanation_service._compute_explanation(
                image_bytes,
                image=image,
                image_batch=v1_input,
                probability=v1_probability,
            )
        )

        explanation_probability = float(
            explanation_result[
                "prediction"
            ]["probability"]
        )

        probability_difference = abs(
            v1_probability
            - explanation_probability
        )

        if probability_difference > 1e-6:
            raise RuntimeError(
                "Prediction/explanation V1 "
                "probability parity failed."
            )

        if (
            primary_label
            != explanation_result[
                "prediction"
            ]["label"]
        ):
            raise RuntimeError(
                "Prediction/explanation V1 "
                "label parity failed."
            )

        return {
            "primary_prediction": prediction[
                "primary_prediction"
            ],
            "secondary_signal": prediction[
                "secondary_signal"
            ],
            "decision": prediction["decision"],
            "preprocessing": prediction[
                "preprocessing"
            ],
            "explanation": explanation_result[
                "explanation"
            ],
            "explanation_quality": (
                explanation_result["quality"]
            ),
            "visualization_endpoints": {
                "heatmap": "/api/v1/explain",
                "overlay": (
                    "/api/v1/explain/overlay"
                ),
            },
            "disclaimer": prediction[
                "disclaimer"
            ],
        }


analysis_service = AnalysisService()
