from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import backend.app.services.explanation_service as explanation_module
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


def configure_explanation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    probability: float | None = None,
    heatmap: np.ndarray | None = None,
    explanation_mode: str = "positive_gradcam",
) -> FakeModel:
    model_probability = (
        inference_service.v1_threshold + 0.10
        if probability is None
        else probability
    )

    model = FakeModel(model_probability)

    monkeypatch.setattr(
        inference_service,
        "load_models",
        lambda: None,
    )
    monkeypatch.setattr(
        inference_service,
        "_v1_model",
        model,
    )
    monkeypatch.setattr(
        inference_service,
        "decode_image",
        lambda image_bytes: np.zeros(
            (8, 10, 3),
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        inference_service,
        "prepare_v1_input",
        lambda image_bytes: "v1-input",
    )
    monkeypatch.setattr(
        explanation_module,
        "find_last_conv_layer",
        lambda model: "conv-test",
    )

    selected_heatmap = (
        np.array(
            [
                [0.0, 0.25],
                [0.50, 1.0],
            ],
            dtype=np.float32,
        )
        if heatmap is None
        else heatmap
    )

    monkeypatch.setattr(
        explanation_module,
        "generate_gradcam_heatmap",
        lambda **kwargs: (
            selected_heatmap,
            explanation_mode,
        ),
    )

    return model


def test_rejects_empty_image() -> None:
    with pytest.raises(
        ValueError,
        match="Uploaded image is empty",
    ):
        explanation_service._compute_explanation(b"")


def test_rejects_missing_primary_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference_service,
        "load_models",
        lambda: None,
    )
    monkeypatch.setattr(
        inference_service,
        "_v1_model",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="Primary V1 model is not loaded",
    ):
        explanation_service._compute_explanation(
            b"test-image"
        )


def test_computes_prediction_and_gradcam_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = configure_explanation(monkeypatch)

    result = explanation_service._compute_explanation(
        b"test-image"
    )

    assert model.calls == 1
    assert model.last_training is False

    assert result["model"] == "baseline_cnn_v1"
    assert result["prediction"]["label"] == (
        "PNEUMONIA"
    )
    assert result["prediction"]["probability"] == (
        pytest.approx(model.probability)
    )

    assert result["explanation"] == {
        "method": "gradcam",
        "mode": "positive_gradcam",
        "last_conv_layer": "conv-test",
        "raw_heatmap_shape": [2, 2],
        "output_width": 10,
        "output_height": 8,
        "minimum": 0.0,
        "maximum": 1.0,
    }

    assert result["heatmap_uint8"].shape == (8, 10)
    assert result["heatmap_uint8"].dtype == np.uint8


def test_reuses_supplied_probability_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = configure_explanation(monkeypatch)

    supplied_probability = max(
        0.0,
        inference_service.v1_threshold - 0.01,
    )

    result = explanation_service._compute_explanation(
        b"test-image",
        image=np.zeros(
            (6, 7, 3),
            dtype=np.uint8,
        ),
        image_batch="existing-v1-input",
        probability=supplied_probability,
    )

    assert model.calls == 0
    assert result["prediction"]["label"] == "NORMAL"
    assert result["prediction"]["probability"] == (
        pytest.approx(supplied_probability)
    )
    assert result["explanation"]["output_width"] == 7
    assert result["explanation"]["output_height"] == 6


def test_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_explanation(monkeypatch)

    result = explanation_service._compute_explanation(
        b"test-image",
        image=np.zeros(
            (8, 10, 3),
            dtype=np.uint8,
        ),
        image_batch="v1-input",
        probability=inference_service.v1_threshold,
    )

    assert result["prediction"]["label"] == (
        "PNEUMONIA"
    )


@pytest.mark.parametrize(
    ("heatmap", "message"),
    [
        (
            np.zeros(
                (2, 2, 1),
                dtype=np.float32,
            ),
            "Unexpected Grad-CAM shape",
        ),
        (
            np.array(
                [
                    [0.0, np.nan],
                    [0.5, 1.0],
                ],
                dtype=np.float32,
            ),
            "Grad-CAM contains invalid values",
        ),
        (
            np.array(
                [
                    [-0.1, 0.0],
                    [0.5, 1.0],
                ],
                dtype=np.float32,
            ),
            "Grad-CAM contains negative values",
        ),
        (
            np.array(
                [
                    [0.0, 0.5],
                    [1.0, 1.01],
                ],
                dtype=np.float32,
            ),
            "Grad-CAM exceeds normalized range",
        ),
    ],
)
def test_rejects_invalid_gradcam_output(
    monkeypatch: pytest.MonkeyPatch,
    heatmap: np.ndarray,
    message: str,
) -> None:
    configure_explanation(
        monkeypatch,
        heatmap=heatmap,
    )

    with pytest.raises(
        RuntimeError,
        match=message,
    ):
        explanation_service._compute_explanation(
            b"test-image"
        )


def test_absolute_attribution_metadata_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_explanation(
        monkeypatch,
        explanation_mode="absolute_attribution",
    )

    result = explanation_service._compute_explanation(
        b"test-image"
    )

    assert result["explanation"]["mode"] == (
        "absolute_attribution"
    )
    assert result["quality"]["explanation_mode"] == (
        "absolute_attribution"
    )
    assert result["quality"]["attribution_note"] is not None


def make_visualization_result(
    image: np.ndarray,
) -> dict[str, Any]:
    return {
        "image": image,
        "heatmap_uint8": np.full(
            image.shape[:2],
            128,
            dtype=np.uint8,
        ),
        "model": "baseline_cnn_v1",
        "prediction": {
            "label": "PNEUMONIA",
            "probability": 0.9,
            "threshold": (
                inference_service.v1_threshold
            ),
        },
        "quality": {
            "quality_status": (
                "LIMITED_SPATIAL_RELIABILITY"
            ),
            "display_warning": True,
            "warning_code": (
                "EXPLANATION_LIMITED_RELIABILITY"
            ),
        },
        "explanation": {
            "method": "gradcam",
            "mode": "positive_gradcam",
            "last_conv_layer": "conv-test",
            "raw_heatmap_shape": [2, 2],
            "output_width": int(image.shape[1]),
            "output_height": int(image.shape[0]),
            "minimum": 0.0,
            "maximum": 1.0,
        },
    }


def test_encode_png_returns_png_bytes() -> None:
    image = np.zeros(
        (4, 5),
        dtype=np.uint8,
    )

    encoded = explanation_service._encode_png(
        image,
        "encoding failed",
    )

    assert isinstance(encoded, bytes)
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_encode_png_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        explanation_module.cv2,
        "imencode",
        lambda *args, **kwargs: (
            False,
            np.array([], dtype=np.uint8),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="custom encoding failure",
    ):
        explanation_service._encode_png(
            np.zeros(
                (2, 2),
                dtype=np.uint8,
            ),
            "custom encoding failure",
        )


def test_explain_bytes_returns_encoded_heatmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_visualization_result(
        np.zeros(
            (6, 7, 3),
            dtype=np.uint8,
        )
    )

    monkeypatch.setattr(
        explanation_service,
        "_compute_explanation",
        lambda image_bytes: result,
    )

    response = explanation_service.explain_bytes(
        b"test-image"
    )

    assert response["model"] == "baseline_cnn_v1"
    assert response["prediction"] == result["prediction"]
    assert response["explanation"] == result["explanation"]
    assert response["quality"] == result["quality"]
    assert response["heatmap_png_bytes"].startswith(
        b"\x89PNG\r\n\x1a\n"
    )


@pytest.mark.parametrize(
    ("image", "expected_shape"),
    [
        (
            np.zeros(
                (6, 7),
                dtype=np.uint8,
            ),
            (6, 7, 3),
        ),
        (
            np.zeros(
                (6, 7, 3),
                dtype=np.uint8,
            ),
            (6, 7, 3),
        ),
        (
            np.zeros(
                (6, 7, 4),
                dtype=np.uint8,
            ),
            (6, 7, 3),
        ),
    ],
    ids=[
        "grayscale",
        "bgr",
        "bgra",
    ],
)
def test_overlay_supports_valid_image_formats(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> None:
    result = make_visualization_result(image)

    monkeypatch.setattr(
        explanation_service,
        "_compute_explanation",
        lambda image_bytes: result,
    )

    captured: dict[str, Any] = {}

    def fake_encode_png(
        encoded_image: np.ndarray,
        error_message: str,
    ) -> bytes:
        captured["shape"] = encoded_image.shape
        captured["error_message"] = error_message
        return b"encoded-overlay"

    monkeypatch.setattr(
        explanation_service,
        "_encode_png",
        fake_encode_png,
    )

    response = explanation_service.overlay_bytes(
        b"test-image"
    )

    assert captured["shape"] == expected_shape
    assert captured["error_message"] == (
        "Could not encode Grad-CAM overlay."
    )

    assert response["overlay_png_bytes"] == (
        b"encoded-overlay"
    )
    assert response["explanation"][
        "visualization"
    ] == "colored_overlay"
    assert response["explanation"]["colormap"] == "jet"
    assert response["explanation"][
        "overlay_alpha"
    ] == pytest.approx(
        explanation_module.OVERLAY_ALPHA
    )

    assert "visualization" not in result["explanation"]


@pytest.mark.parametrize(
    "image",
    [
        np.zeros(
            (6, 7, 2),
            dtype=np.uint8,
        ),
        np.zeros(
            (6, 7, 5),
            dtype=np.uint8,
        ),
    ],
)
def test_overlay_rejects_unsupported_image_format(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
) -> None:
    result = make_visualization_result(image)

    monkeypatch.setattr(
        explanation_service,
        "_compute_explanation",
        lambda image_bytes: result,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported image format "
            "for overlay generation"
        ),
    ):
        explanation_service.overlay_bytes(
            b"test-image"
        )
