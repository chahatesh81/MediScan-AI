from fastapi import APIRouter

from backend.app.schemas.prediction import (
    DeploymentPolicyInfo,
    ModelInfoResponse,
    PrimaryModelInfo,
    SecondaryModelInfo,
)
from backend.app.services.inference_service import (
    inference_service,
)


router = APIRouter()


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Models"],
)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        primary_model=PrimaryModelInfo(
            name="baseline_cnn_v1",
            role="primary_classifier",
            threshold=(
                inference_service.v1_threshold
            ),
        ),
        secondary_model=SecondaryModelInfo(
            name="advanced_v3",
            role="exploratory_safety_signal",
            threshold=(
                inference_service.v3_threshold
            ),
            automatic_override_allowed=False,
        ),
        deployment_policy=(
            DeploymentPolicyInfo(
                primary_prediction_source=(
                    "baseline_cnn_v1"
                ),
                automatic_override_allowed=False,
                automatic_ensemble_allowed=False,
                warning_condition=(
                    inference_service
                    .warning_condition
                ),
            )
        ),
        disclaimer=(
            "Educational decision-support prototype. "
            "Not for clinical use. Human review required."
        ),
    )
