from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CropBoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class V3PreprocessingMetadata(BaseModel):
    original_width: int
    original_height: int
    crop_bbox: CropBoundingBox
    cropped_width: int
    cropped_height: int
    retained_area_ratio: float
    target_size: int


class PrimaryPrediction(BaseModel):
    model: Literal["baseline_cnn_v1"]
    label: Literal[
        "NORMAL",
        "PNEUMONIA",
    ]
    probability: float
    threshold: float


class SecondarySignal(BaseModel):
    model: Literal["advanced_v3"]
    role: Literal["exploratory"]
    probability: float
    threshold: float
    predicted_label: Literal[
        "NORMAL",
        "PNEUMONIA",
    ]
    automatic_override_allowed: bool


class Decision(BaseModel):
    final_label: Literal[
        "NORMAL",
        "PNEUMONIA",
    ]
    source: Literal["baseline_cnn_v1"]
    manual_review_recommended: bool
    warning_code: (
        Literal["V1_NORMAL_V3_PNEUMONIA"]
        | None
    )


class PreprocessingInfo(BaseModel):
    v1: Literal[
        "rgb_bilinear_resize_224"
    ]
    v3: Literal[
        "artifact_aware_preprocess_xray"
    ]
    v3_metadata: V3PreprocessingMetadata


class PredictionResponse(BaseModel):
    primary_prediction: PrimaryPrediction
    secondary_signal: SecondarySignal
    decision: Decision
    preprocessing: PreprocessingInfo
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal[
        "ready",
        "not_ready",
    ]
    service: str
    models_loaded: bool


class PrimaryModelInfo(BaseModel):
    name: str
    role: str
    threshold: float


class SecondaryModelInfo(BaseModel):
    name: str
    role: str
    threshold: float
    automatic_override_allowed: bool


class DeploymentPolicyInfo(BaseModel):
    primary_prediction_source: str
    automatic_override_allowed: bool
    automatic_ensemble_allowed: bool
    warning_condition: str


class ModelInfoResponse(BaseModel):
    primary_model: PrimaryModelInfo
    secondary_model: SecondaryModelInfo
    deployment_policy: DeploymentPolicyInfo
    disclaimer: str
