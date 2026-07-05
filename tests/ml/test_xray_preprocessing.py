from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.app.ml.xray_preprocessing import (
    build_foreground_mask,
    crop_foreground,
    ensure_grayscale,
    estimate_background_intensity,
    find_foreground_bbox,
    normalize_to_uint8,
    preprocess_xray,
    resize_with_padding,
)


def test_ensure_grayscale_rejects_none() -> None:
    with pytest.raises(
        ValueError,
        match="Input image is None",
    ):
        ensure_grayscale(None)


def test_ensure_grayscale_rejects_empty_image() -> None:
    image = np.empty(
        (0, 0),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Input image is empty",
    ):
        ensure_grayscale(image)


def test_ensure_grayscale_preserves_2d_image() -> None:
    image = np.arange(
        16,
        dtype=np.uint8,
    ).reshape(4, 4)

    result = ensure_grayscale(image)

    np.testing.assert_array_equal(
        result,
        image,
    )


def test_ensure_grayscale_squeezes_single_channel() -> None:
    image = np.arange(
        16,
        dtype=np.uint8,
    ).reshape(4, 4, 1)

    result = ensure_grayscale(image)

    assert result.shape == (4, 4)
    np.testing.assert_array_equal(
        result,
        image[..., 0],
    )


def test_ensure_grayscale_converts_bgr_image() -> None:
    image = np.zeros(
        (4, 4, 3),
        dtype=np.uint8,
    )
    image[..., 2] = 255

    result = ensure_grayscale(image)

    expected = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_ensure_grayscale_converts_bgra_image() -> None:
    image = np.zeros(
        (4, 4, 4),
        dtype=np.uint8,
    )
    image[..., 1] = 200
    image[..., 3] = 255

    result = ensure_grayscale(image)

    expected = cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2GRAY,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_ensure_grayscale_rejects_unsupported_channels() -> None:
    image = np.zeros(
        (4, 4, 2),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported channel count: 2",
    ):
        ensure_grayscale(image)


def test_ensure_grayscale_rejects_invalid_dimensions() -> None:
    image = np.zeros(
        (2, 2, 2, 2),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Expected a 2D or 3D image",
    ):
        ensure_grayscale(image)


def test_normalize_uint8_returns_independent_copy() -> None:
    image = np.array(
        [
            [0, 64],
            [128, 255],
        ],
        dtype=np.uint8,
    )

    result = normalize_to_uint8(image)

    assert result.dtype == np.uint8
    assert result is not image

    result[0, 0] = 255

    assert image[0, 0] == 0


def test_normalize_float_image_uses_full_uint8_range() -> None:
    image = np.array(
        [
            [-10.0, 0.0],
            [5.0, 10.0],
        ],
        dtype=np.float32,
    )

    result = normalize_to_uint8(image)

    assert result.dtype == np.uint8
    assert int(result.min()) == 0
    assert int(result.max()) == 255


def test_normalize_constant_image_returns_zeros() -> None:
    image = np.full(
        (5, 7),
        42.0,
        dtype=np.float32,
    )

    result = normalize_to_uint8(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.count_nonzero(result) == 0


def test_normalize_rejects_image_without_finite_values() -> None:
    image = np.array(
        [
            [np.nan, np.inf],
            [-np.inf, np.nan],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Image contains no finite values",
    ):
        normalize_to_uint8(image)


def test_estimate_background_uses_corner_intensity() -> None:
    image = np.full(
        (100, 100),
        20,
        dtype=np.uint8,
    )
    image[20:80, 20:80] = 200

    background = estimate_background_intensity(
        image
    )

    assert background == pytest.approx(20.0)


def test_foreground_mask_matches_input_geometry() -> None:
    image = np.zeros(
        (120, 160),
        dtype=np.uint8,
    )
    image[20:100, 30:130] = 180

    mask = build_foreground_mask(image)

    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset(
        {0, 255}
    )


def test_uniform_image_falls_back_to_full_bbox() -> None:
    image = np.full(
        (100, 140),
        50,
        dtype=np.uint8,
    )

    bbox = find_foreground_bbox(image)

    assert bbox == (0, 0, 140, 100)


def test_small_foreground_falls_back_to_full_bbox() -> None:
    image = np.zeros(
        (200, 200),
        dtype=np.uint8,
    )
    image[90:110, 90:110] = 255

    bbox = find_foreground_bbox(image)

    assert bbox == (0, 0, 200, 200)


def test_large_foreground_produces_safe_crop() -> None:
    image = np.zeros(
        (200, 200),
        dtype=np.uint8,
    )
    image[30:170, 40:160] = 200

    x_min, y_min, x_max, y_max = (
        find_foreground_bbox(image)
    )

    assert 0 <= x_min < x_max <= 200
    assert 0 <= y_min < y_max <= 200
    assert x_min <= 40
    assert y_min <= 30
    assert x_max >= 160
    assert y_max >= 170
    assert (
        x_min,
        y_min,
        x_max,
        y_max,
    ) != (0, 0, 200, 200)


def test_crop_foreground_returns_crop_and_matching_bbox() -> None:
    image = np.zeros(
        (200, 200),
        dtype=np.uint8,
    )
    image[30:170, 40:160] = 200

    cropped, bbox = crop_foreground(image)

    x_min, y_min, x_max, y_max = bbox

    assert cropped.shape == (
        y_max - y_min,
        x_max - x_min,
    )


@pytest.mark.parametrize(
    "target_size",
    [0, -1],
)
def test_resize_rejects_non_positive_target(
    target_size: int,
) -> None:
    image = np.ones(
        (10, 10),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="target_size must be positive",
    ):
        resize_with_padding(
            image,
            target_size=target_size,
        )


def test_resize_with_padding_returns_square_uint8() -> None:
    image = np.full(
        (50, 100),
        40,
        dtype=np.uint8,
    )
    image[10:40, 20:80] = 180

    result = resize_with_padding(
        image,
        target_size=224,
    )

    assert result.shape == (224, 224)
    assert result.dtype == np.uint8


def test_resize_with_padding_preserves_aspect_ratio() -> None:
    image = np.zeros(
        (50, 100),
        dtype=np.uint8,
    )
    image[5:45, 10:90] = 200

    result = resize_with_padding(
        image,
        target_size=200,
    )

    assert result.shape == (200, 200)
    assert np.all(result[:50] == 0)
    assert np.any(result[50:150] != 0)
    assert np.all(result[150:] == 0)


def test_preprocess_returns_rgb_uint8_contract() -> None:
    image = np.zeros(
        (180, 240),
        dtype=np.uint16,
    )
    image[20:160, 30:210] = 4095

    result = preprocess_xray(
        image,
        target_size=128,
    )

    assert result.shape == (128, 128, 3)
    assert result.dtype == np.uint8

    np.testing.assert_array_equal(
        result[..., 0],
        result[..., 1],
    )
    np.testing.assert_array_equal(
        result[..., 1],
        result[..., 2],
    )


def test_preprocess_metadata_matches_output_and_crop() -> None:
    image = np.zeros(
        (200, 300),
        dtype=np.uint8,
    )
    image[20:180, 40:260] = 200

    processed, metadata = preprocess_xray(
        image,
        target_size=96,
        return_metadata=True,
    )

    bbox = metadata["crop_bbox"]

    assert processed.shape == (96, 96, 3)
    assert metadata["original_width"] == 300
    assert metadata["original_height"] == 200
    assert metadata["target_size"] == 96
    assert metadata["cropped_width"] == (
        bbox["x_max"] - bbox["x_min"]
    )
    assert metadata["cropped_height"] == (
        bbox["y_max"] - bbox["y_min"]
    )
    assert (
        0.0
        < metadata["retained_area_ratio"]
        <= 1.0
    )


def test_preprocess_is_deterministic() -> None:
    rng = np.random.default_rng(12345)

    image = rng.integers(
        0,
        256,
        size=(180, 220),
        dtype=np.uint8,
    )

    first = preprocess_xray(
        image,
        target_size=128,
    )
    second = preprocess_xray(
        image,
        target_size=128,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_preprocess_does_not_mutate_input() -> None:
    image = np.zeros(
        (120, 160, 3),
        dtype=np.uint8,
    )
    image[20:100, 30:130] = (
        20,
        100,
        220,
    )

    original = image.copy()

    preprocess_xray(
        image,
        target_size=64,
    )

    np.testing.assert_array_equal(
        image,
        original,
    )
