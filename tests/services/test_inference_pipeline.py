from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import tensorflow as tf

from backend.app.core.config import IMAGE_SIZE
from backend.app.services import inference_service as inference_module
from backend.app.services.inference_service import (
    inference_service,
)


pytestmark = pytest.mark.unit


def encode_image(
    image: np.ndarray,
    extension: str = ".png",
) -> bytes:
    success, encoded = cv2.imencode(
        extension,
        image,
    )

    assert success is True

    return encoded.tobytes()


@pytest.mark.parametrize(
    (
        "image",
        "expected_shape",
    ),
    [
        (
            np.full(
                (24, 32),
                127,
                dtype=np.uint8,
            ),
            (24, 32),
        ),
        (
            np.full(
                (24, 32, 3),
                127,
                dtype=np.uint8,
            ),
            (24, 32, 3),
        ),
        (
            np.full(
                (24, 32, 4),
                127,
                dtype=np.uint8,
            ),
            (24, 32, 4),
        ),
    ],
)
def test_decode_image_accepts_supported_image_shapes(
    image: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    result = inference_service.decode_image(
        encode_image(image)
    )

    assert result.shape == expected_shape
    assert result.dtype == np.uint8


def test_decode_image_rejects_empty_upload() -> None:
    with pytest.raises(
        ValueError,
        match="Uploaded image is empty",
    ):
        inference_service.decode_image(b"")


def test_decode_image_rejects_invalid_image_bytes() -> None:
    with pytest.raises(
        ValueError,
        match="not a valid image",
    ):
        inference_service.decode_image(
            b"not-an-image"
        )


def test_decode_image_rejects_unsupported_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference_module.cv2,
        "imdecode",
        lambda encoded, flags: np.zeros(
            (2, 3, 4, 5),
            dtype=np.uint8,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported image dimensions",
    ):
        inference_service.decode_image(
            b"encoded-image"
        )


def test_prepare_v1_input_returns_frozen_tensor_contract() -> None:
    image = np.zeros(
        (40, 80, 3),
        dtype=np.uint8,
    )
    image[:, :, 0] = 10
    image[:, :, 1] = 100
    image[:, :, 2] = 240

    image_bytes = encode_image(
        image,
        extension=".jpg",
    )

    result = inference_service.prepare_v1_input(
        image_bytes
    )

    assert isinstance(result, tf.Tensor)
    assert result.shape == (
        1,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    )
    assert result.dtype == tf.float32

    values = result.numpy()

    assert float(values.min()) >= 0.0
    assert float(values.max()) > 1.0
    assert float(values.max()) <= 255.0


def test_prepare_v1_input_rejects_non_jpeg_bytes() -> None:
    with pytest.raises(
        tf.errors.InvalidArgumentError,
    ):
        inference_service.prepare_v1_input(
            b"not-a-jpeg"
        )


def test_prepare_v3_input_returns_tensor_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.full(
        (40, 80, 3),
        100,
        dtype=np.uint8,
    )

    processed = np.full(
        (
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        180,
        dtype=np.uint8,
    )

    metadata = {
        "pipeline": "test-preprocessing",
        "crop_bbox": [1, 2, 30, 40],
    }

    captured: dict[str, Any] = {}

    def fake_preprocess_xray(
        image: np.ndarray,
        *,
        target_size: int,
        return_metadata: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        captured["image"] = image
        captured["target_size"] = target_size
        captured["return_metadata"] = (
            return_metadata
        )

        return processed, metadata

    monkeypatch.setattr(
        inference_module,
        "preprocess_xray",
        fake_preprocess_xray,
    )

    result, result_metadata = (
        inference_service.prepare_v3_input(
            source
        )
    )

    assert captured["image"] is source
    assert captured["target_size"] == (
        IMAGE_SIZE[0]
    )
    assert captured["return_metadata"] is True

    assert isinstance(result, tf.Tensor)
    assert result.shape == (
        1,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    )
    assert result.dtype == tf.float32
    assert result_metadata is metadata


def test_prepare_v3_input_uses_jpeg_quality_95(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.zeros(
        (32, 48, 3),
        dtype=np.uint8,
    )

    processed = np.full(
        (
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        150,
        dtype=np.uint8,
    )

    real_imencode = cv2.imencode
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        inference_module,
        "preprocess_xray",
        lambda image, target_size, return_metadata: (
            processed,
            {"pipeline": "test"},
        ),
    )

    def recording_imencode(
        extension: str,
        image: np.ndarray,
        params: list[int],
    ) -> tuple[bool, np.ndarray]:
        captured["extension"] = extension
        captured["image"] = image
        captured["params"] = params

        return real_imencode(
            extension,
            image,
            params,
        )

    monkeypatch.setattr(
        inference_module.cv2,
        "imencode",
        recording_imencode,
    )

    inference_service.prepare_v3_input(source)

    assert captured["extension"] == ".jpg"
    assert captured["image"] is processed
    assert captured["params"] == [
        cv2.IMWRITE_JPEG_QUALITY,
        95,
    ]


def test_prepare_v3_input_rejects_jpeg_encoding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = np.zeros(
        (
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        inference_module,
        "preprocess_xray",
        lambda image, target_size, return_metadata: (
            processed,
            {},
        ),
    )

    monkeypatch.setattr(
        inference_module.cv2,
        "imencode",
        lambda extension, image, params: (
            False,
            np.array([], dtype=np.uint8),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Could not reproduce V3 cached JPEG preprocessing",
    ):
        inference_service.prepare_v3_input(
            np.zeros(
                (20, 20, 3),
                dtype=np.uint8,
            )
        )


def test_load_models_loads_each_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_model = object()
    v3_model = object()

    calls: list[tuple[Path, bool]] = []

    def fake_load_model(
        path: Path,
        *,
        compile: bool,
    ) -> object:
        calls.append((path, compile))

        if path == inference_service.v1_model_path:
            return v1_model

        if path == inference_service.v3_model_path:
            return v3_model

        raise AssertionError(
            f"Unexpected model path: {path}"
        )

    monkeypatch.setattr(
        inference_service,
        "_v1_model",
        None,
    )
    monkeypatch.setattr(
        inference_service,
        "_v3_model",
        None,
    )
    monkeypatch.setattr(
        inference_module.tf.keras.models,
        "load_model",
        fake_load_model,
    )

    inference_service.load_models()
    inference_service.load_models()

    assert inference_service._v1_model is v1_model
    assert inference_service._v3_model is v3_model

    assert calls == [
        (
            inference_service.v1_model_path,
            False,
        ),
        (
            inference_service.v3_model_path,
            False,
        ),
    ]


@pytest.mark.parametrize(
    "preloaded_model",
    [
        "v1",
        "v3",
    ],
)
def test_load_models_recovers_from_partial_loaded_state(
    monkeypatch: pytest.MonkeyPatch,
    preloaded_model: str,
) -> None:
    existing_model = object()
    loaded_model = object()

    calls: list[Path] = []

    if preloaded_model == "v1":
        monkeypatch.setattr(
            inference_service,
            "_v1_model",
            existing_model,
        )
        monkeypatch.setattr(
            inference_service,
            "_v3_model",
            None,
        )
        expected_path = (
            inference_service.v3_model_path
        )
    else:
        monkeypatch.setattr(
            inference_service,
            "_v1_model",
            None,
        )
        monkeypatch.setattr(
            inference_service,
            "_v3_model",
            existing_model,
        )
        expected_path = (
            inference_service.v1_model_path
        )

    def fake_load_model(
        path: Path,
        *,
        compile: bool,
    ) -> object:
        calls.append(path)

        assert compile is False

        return loaded_model

    monkeypatch.setattr(
        inference_module.tf.keras.models,
        "load_model",
        fake_load_model,
    )

    inference_service.load_models()

    assert calls == [expected_path]

    if preloaded_model == "v1":
        assert (
            inference_service._v1_model
            is existing_model
        )
        assert (
            inference_service._v3_model
            is loaded_model
        )
    else:
        assert (
            inference_service._v1_model
            is loaded_model
        )
        assert (
            inference_service._v3_model
            is existing_model
        )


def test_load_models_returns_without_loader_when_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_model = object()
    v3_model = object()

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

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "load_model must not be called "
            "when both models are already loaded."
        )

    monkeypatch.setattr(
        inference_module.tf.keras.models,
        "load_model",
        fail_if_called,
    )

    inference_service.load_models()

    assert inference_service._v1_model is v1_model
    assert inference_service._v3_model is v3_model
