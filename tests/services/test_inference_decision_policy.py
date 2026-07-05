from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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


def configure_inference(
    monkeypatch: pytest.MonkeyPatch,
    *,
    v1_probability: float,
    v3_probability: float,
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
                "pipeline": "test-v3",
            },
        ),
    )

    return v1_model, v3_model


@pytest.mark.parametrize(
    (
        "v1_probability",
        "v3_probability",
        "expected_primary",
        "expected_secondary",
        "manual_review",
        "warning_code",
    ),
    [
        (
            inference_service.v1_threshold + 0.10,
            inference_service.v3_threshold + 0.10,
            "PNEUMONIA",
            "PNEUMONIA",
            False,
            None,
        ),
        (
            inference_service.v1_threshold + 0.10,
            max(0.0, inference_service.v3_threshold - 0.10),
            "PNEUMONIA",
            "NORMAL",
            False,
            None,
        ),
        (
            max(0.0, inference_service.v1_threshold - 0.01),
            inference_service.v3_threshold + 0.10,
            "NORMAL",
            "PNEUMONIA",
            True,
            "V1_NORMAL_V3_PNEUMONIA",
        ),
        (
            max(0.0, inference_service.v1_threshold - 0.01),
            max(0.0, inference_service.v3_threshold - 0.10),
            "NORMAL",
            "NORMAL",
            False,
            None,
        ),
    ],
)
def test_complete_v1_v3_decision_matrix(
    monkeypatch: pytest.MonkeyPatch,
    v1_probability: float,
    v3_probability: float,
    expected_primary: str,
    expected_secondary: str,
    manual_review: bool,
    warning_code: str | None,
) -> None:
    v1_model, v3_model = configure_inference(
        monkeypatch,
        v1_probability=v1_probability,
        v3_probability=v3_probability,
    )

    result = inference_service.predict_bytes(
        b"test-image"
    )

    assert result["primary_prediction"]["label"] == (
        expected_primary
    )
    assert result["secondary_signal"][
        "predicted_label"
    ] == expected_secondary

    assert result["decision"]["final_label"] == (
        expected_primary
    )
    assert result["decision"]["source"] == (
        "baseline_cnn_v1"
    )
    assert result["decision"][
        "manual_review_recommended"
    ] is manual_review
    assert result["decision"]["warning_code"] == (
        warning_code
    )

    assert result["secondary_signal"][
        "automatic_override_allowed"
    ] is False

    assert v1_model.calls == 1
    assert v3_model.calls == 1
    assert v1_model.last_training is False
    assert v3_model.last_training is False


def test_threshold_values_are_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inference(
        monkeypatch,
        v1_probability=inference_service.v1_threshold,
        v3_probability=inference_service.v3_threshold,
    )

    result = inference_service.predict_bytes(
        b"test-image"
    )

    assert result["primary_prediction"]["label"] == (
        "PNEUMONIA"
    )
    assert result["secondary_signal"][
        "predicted_label"
    ] == "PNEUMONIA"
    assert result["decision"][
        "manual_review_recommended"
    ] is False


def test_v3_never_overrides_v1_normal_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inference(
        monkeypatch,
        v1_probability=0.0,
        v3_probability=1.0,
    )

    result = inference_service.predict_bytes(
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


def test_preprocessing_metadata_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_inference(
        monkeypatch,
        v1_probability=0.9,
        v3_probability=0.2,
    )

    result = inference_service.predict_bytes(
        b"test-image"
    )

    assert result["preprocessing"]["v1"] == (
        "rgb_bilinear_resize_224"
    )
    assert result["preprocessing"]["v3"] == (
        "artifact_aware_preprocess_xray"
    )
    assert result["preprocessing"]["v3_metadata"] == {
        "pipeline": "test-v3",
    }


@pytest.mark.parametrize(
    ("missing_model", "expected_attribute"),
    [
        ("v1", "_v1_model"),
        ("v3", "_v3_model"),
    ],
)
def test_predict_rejects_missing_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
    missing_model: str,
    expected_attribute: str,
) -> None:
    v1_model = FakeModel(0.9)
    v3_model = FakeModel(0.9)

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
        expected_attribute,
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="Models failed to load",
    ):
        inference_service.predict_bytes(
            b"test-image"
        )
