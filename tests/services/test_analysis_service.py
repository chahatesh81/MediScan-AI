from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from backend.app.services.analysis_service import (
    analysis_service,
)
from backend.app.services.explanation_service import (
    explanation_service,
)
from backend.app.services.inference_service import (
    inference_service,
)


pytestmark = pytest.mark.unit


class FakeTensor:
    def __init__(self, value: float) -> None:
        self._value = value

    def numpy(self) -> np.ndarray:
        return np.array(
            [[self._value]],
            dtype=np.float32,
        )


class FakeModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls = 0
        self.last_training: bool | None = None

    def __call__(
        self,
        image_batch: Any,
        *,
        training: bool,
    ) -> FakeTensor:
        self.calls += 1
        self.last_training = training
        return FakeTensor(self.probability)


def configure_analysis(
    monkeypatch: pytest.MonkeyPatch,
    *,
    v1_probability: float,
    v3_probability: float,
    explanation_probability: float | None = None,
    explanation_label: str | None = None,
) -> tuple[FakeModel, FakeModel]:
    v1_model = FakeModel(v1_probability)
    v3_model = FakeModel(v3_probability)

    monkeypatch.setattr(
        inference_service,
        "load_models",
        lambda: None,
    )
    monkeypatch.setattr(
        inference_service,
        "_v1_model",
        v1_model,
    )
    monkeypatch.setattr(
        inference_service,
        "_v3_model",
        v3_model,
    )
    monkeypatch.setattr(
        inference_service,
        "decode_image",
        lambda image_bytes: np.zeros(
            (8, 8, 3),
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        inference_service,
        "prepare_v1_input",
        lambda image_bytes: "v1-input",
    )
    monkeypatch.setattr(
        inference_service,
        "prepare_v3_input",
        lambda image: (
            "v3-input",
            {
                "original_width": 8,
                "original_height": 8,
                "crop_bbox": {
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 8,
                    "y_max": 8,
                },
                "cropped_width": 8,
                "cropped_height": 8,
                "retained_area_ratio": 1.0,
                "target_size": 224,
            },
        ),
    )

    expected_label = (
        "PNEUMONIA"
        if v1_probability
        >= inference_service.v1_threshold
        else "NORMAL"
    )

    returned_probability = (
        v1_probability
        if explanation_probability is None
        else explanation_probability
    )

    returned_label = (
        expected_label
        if explanation_label is None
        else explanation_label
    )

    def fake_explanation(
        image_bytes: bytes,
        *,
        image: np.ndarray,
        image_batch: Any,
        probability: float,
    ) -> dict[str, Any]:
        assert image_bytes == b"test-image"
        assert image.shape == (8, 8, 3)
        assert image_batch == "v1-input"
        assert probability == pytest.approx(
            v1_probability
        )

        return {
            "prediction": {
                "label": returned_label,
                "probability": returned_probability,
                "threshold": (
                    inference_service.v1_threshold
                ),
            },
            "explanation": {
                "method": "gradcam",
                "mode": "positive_gradcam",
                "last_conv_layer": "conv-test",
                "raw_heatmap_shape": [4, 4],
                "output_width": 8,
                "output_height": 8,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "quality": {
                "quality_status": (
                    "LIMITED_SPATIAL_RELIABILITY"
                ),
                "display_warning": True,
                "warning_code": (
                    "EXPLANATION_LIMITED_RELIABILITY"
                ),
                "border_energy_ratio": 0.2,
                "thorax_energy_ratio": 0.8,
                "peak_in_border": 0.0,
                "explanation_mode": (
                    "positive_gradcam"
                ),
                "attribution_note": None,
                "region_definition": (
                    "geometric_proxy_not_anatomical_lung_mask"
                ),
            },
        }

    monkeypatch.setattr(
        explanation_service,
        "_compute_explanation",
        fake_explanation,
    )

    return v1_model, v3_model


def test_combined_analysis_reuses_one_v1_forward_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_model, v3_model = configure_analysis(
        monkeypatch,
        v1_probability=(
            inference_service.v1_threshold + 0.10
        ),
        v3_probability=(
            inference_service.v3_threshold - 0.10
        ),
    )

    result = analysis_service.analyze_bytes(
        b"test-image"
    )

    assert v1_model.calls == 1
    assert v3_model.calls == 1
    assert v1_model.last_training is False
    assert v3_model.last_training is False

    assert result["decision"] == {
        "final_label": "PNEUMONIA",
        "source": "baseline_cnn_v1",
        "manual_review_recommended": False,
        "warning_code": None,
    }

    assert result["explanation"]["method"] == (
        "gradcam"
    )
    assert result["visualization_endpoints"] == {
        "heatmap": "/api/v1/explain",
        "overlay": "/api/v1/explain/overlay",
    }


def test_analysis_preserves_manual_review_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_analysis(
        monkeypatch,
        v1_probability=max(
            0.0,
            inference_service.v1_threshold - 0.01,
        ),
        v3_probability=(
            inference_service.v3_threshold + 0.10
        ),
    )

    result = analysis_service.analyze_bytes(
        b"test-image"
    )

    assert result["primary_prediction"]["label"] == (
        "NORMAL"
    )
    assert result["secondary_signal"][
        "predicted_label"
    ] == "PNEUMONIA"

    assert result["decision"] == {
        "final_label": "NORMAL",
        "source": "baseline_cnn_v1",
        "manual_review_recommended": True,
        "warning_code": (
            "V1_NORMAL_V3_PNEUMONIA"
        ),
    }


def test_explanation_quality_never_changes_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_analysis(
        monkeypatch,
        v1_probability=max(
            0.0,
            inference_service.v1_threshold - 0.01,
        ),
        v3_probability=max(
            0.0,
            inference_service.v3_threshold - 0.10,
        ),
    )

    result = analysis_service.analyze_bytes(
        b"test-image"
    )

    assert result["decision"]["final_label"] == (
        "NORMAL"
    )
    assert result["explanation_quality"][
        "display_warning"
    ] is True
    assert result["decision"][
        "manual_review_recommended"
    ] is False


def test_probability_parity_failure_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_probability = (
        inference_service.v1_threshold + 0.10
    )

    configure_analysis(
        monkeypatch,
        v1_probability=v1_probability,
        v3_probability=(
            inference_service.v3_threshold
        ),
        explanation_probability=(
            v1_probability + 0.01
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Prediction/explanation V1 "
            "probability parity failed"
        ),
    ):
        analysis_service.analyze_bytes(
            b"test-image"
        )


def test_label_parity_failure_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_analysis(
        monkeypatch,
        v1_probability=(
            inference_service.v1_threshold + 0.10
        ),
        v3_probability=(
            inference_service.v3_threshold
        ),
        explanation_label="NORMAL",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Prediction/explanation V1 "
            "label parity failed"
        ),
    ):
        analysis_service.analyze_bytes(
            b"test-image"
        )


@pytest.mark.parametrize(
    "missing_attribute",
    [
        "_v1_model",
        "_v3_model",
    ],
)
def test_analysis_rejects_missing_model(
    monkeypatch: pytest.MonkeyPatch,
    missing_attribute: str,
) -> None:
    monkeypatch.setattr(
        inference_service,
        "load_models",
        lambda: None,
    )
    monkeypatch.setattr(
        inference_service,
        "_v1_model",
        FakeModel(0.9),
    )
    monkeypatch.setattr(
        inference_service,
        "_v3_model",
        FakeModel(0.9),
    )
    monkeypatch.setattr(
        inference_service,
        missing_attribute,
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="Models failed to load",
    ):
        analysis_service.analyze_bytes(
            b"test-image"
        )


def test_analysis_result_validates_complete_response_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.schemas.analysis import AnalysisResponse

    configure_analysis(
        monkeypatch,
        v1_probability=(
            inference_service.v1_threshold + 0.10
        ),
        v3_probability=(
            inference_service.v3_threshold - 0.10
        ),
    )

    result = analysis_service.analyze_bytes(
        b"test-image"
    )

    response = AnalysisResponse(**result)
    payload = response.model_dump(mode="json")

    assert payload["primary_prediction"]["label"] == (
        "PNEUMONIA"
    )
    assert payload["decision"]["final_label"] == (
        "PNEUMONIA"
    )
    assert payload["decision"]["source"] == (
        "baseline_cnn_v1"
    )
    assert payload["secondary_signal"][
        "automatic_override_allowed"
    ] is False
    assert payload["explanation_quality"][
        "peak_in_border"
    ] is False
    assert isinstance(
        payload["explanation_quality"]["peak_in_border"],
        bool,
    )


def test_analysis_contract_preserves_disagreement_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.schemas.analysis import AnalysisResponse

    configure_analysis(
        monkeypatch,
        v1_probability=max(
            0.0,
            inference_service.v1_threshold - 0.01,
        ),
        v3_probability=(
            inference_service.v3_threshold + 0.10
        ),
    )

    result = analysis_service.analyze_bytes(
        b"test-image"
    )

    response = AnalysisResponse(**result)

    assert response.primary_prediction.label == "NORMAL"
    assert (
        response.secondary_signal.predicted_label
        == "PNEUMONIA"
    )
    assert response.decision.final_label == "NORMAL"
    assert response.decision.source == "baseline_cnn_v1"
    assert (
        response.decision.manual_review_recommended
        is True
    )
    assert response.decision.warning_code == (
        "V1_NORMAL_V3_PNEUMONIA"
    )
    assert (
        response.secondary_signal.automatic_override_allowed
        is False
    )


def test_analysis_contract_preserves_preprocessing_and_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.schemas.analysis import AnalysisResponse

    configure_analysis(
        monkeypatch,
        v1_probability=(
            inference_service.v1_threshold + 0.10
        ),
        v3_probability=(
            inference_service.v3_threshold
        ),
    )

    result = analysis_service.analyze_bytes(
        b"test-image"
    )

    response = AnalysisResponse(**result)

    assert response.preprocessing.v1 == (
        "rgb_bilinear_resize_224"
    )
    assert response.preprocessing.v3 == (
        "artifact_aware_preprocess_xray"
    )
    assert response.explanation.method == "gradcam"
    assert response.explanation.mode == (
        "positive_gradcam"
    )
    assert response.explanation_quality.display_warning is True
    assert response.visualization_endpoints.heatmap == (
        "/api/v1/explain"
    )
    assert response.visualization_endpoints.overlay == (
        "/api/v1/explain/overlay"
    )
