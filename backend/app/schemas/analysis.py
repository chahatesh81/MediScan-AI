from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from backend.app.schemas.prediction import (
    Decision,
    PreprocessingInfo,
    PrimaryPrediction,
    SecondarySignal,
)


class ExplanationInfo(BaseModel):
    method: Literal["gradcam"]
    mode: Literal[
        "positive_gradcam",
        "absolute_attribution",
    ]
    last_conv_layer: str
    raw_heatmap_shape: list[int]
    output_width: int
    output_height: int
    minimum: float
    maximum: float


class AttentionQuality(BaseModel):
    border_energy_ratio: float
    thorax_energy_ratio: float
    peak_in_border: float
    quality_status: Literal[
        "HIGH_SHORTCUT_RISK",
        "ELEVATED_SHORTCUT_RISK",
        "LIMITED_SPATIAL_RELIABILITY",
    ]
    display_warning: bool
    warning_code: str | None
    explanation_mode: Literal[
        "positive_gradcam",
        "absolute_attribution",
    ]
    attribution_note: str | None
    region_definition: str


class VisualizationEndpoints(BaseModel):
    heatmap: Literal["/api/v1/explain"]
    overlay: Literal["/api/v1/explain/overlay"]


class AnalysisResponse(BaseModel):
    primary_prediction: PrimaryPrediction
    secondary_signal: SecondarySignal
    decision: Decision
    preprocessing: PreprocessingInfo
    explanation: ExplanationInfo
    explanation_quality: AttentionQuality
    visualization_endpoints: VisualizationEndpoints
    disclaimer: str
